"""
clustering.py — Pose clustering via ProLIF PLIF fingerprints.

Clustering is on PLIF interaction fingerprints, NOT Cartesian RMSD.
RMSD clustering across different receptor conformers is mathematically
incoherent — coordinate frames differ between conformers.
PLIF clustering compares pharmacophoric equivalence, which is the
scientifically correct and chemically meaningful similarity metric here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize
except ImportError as e:
    raise ImportError("scikit-learn is required: pip install scikit-learn") from e

try:
    import prody
    prody.confProDy(verbosity="none")
except ImportError as e:
    raise ImportError("ProDy is required: conda install -c conda-forge prody") from e

from spade.core.docking import DockingResult, PoseResult


@dataclass
class PoseCluster:
    """A cluster of docked poses grouped by PLIF similarity."""
    cluster_id: int
    representative_pose: PoseResult
    member_poses: list[PoseResult]
    n_conformers_represented: int  # how many distinct conformers have poses here
    fraction_ensemble: float       # n_conformers_represented / total_conformers
    mean_score: float              # mean Vina score across all members
    score_std: float               # std of Vina scores
    consensus_score: float         # mean_score weighted by fraction_ensemble
    interaction_fingerprint: np.ndarray  # mean PLIF vector


@dataclass
class ConsensusResult:
    """Top-level output of pose clustering."""
    top_cluster: PoseCluster
    all_clusters: list[PoseCluster]
    pose_confidence: float    # fraction of poses in the top cluster
    site_confidence: str      # "high" / "medium" / "low" based on score_std
    n_total_poses: int
    n_clusters: int


def cluster_poses(
    docking_results: list[DockingResult],
    conformers: list["prody.AtomGroup"],
    ligand_mol,
    similarity_threshold: float = 0.7,
) -> ConsensusResult:
    """
    Cluster docked poses by PLIF interaction fingerprint similarity.

    # NOTE: Clustering is on PLIF interaction fingerprints, NOT Cartesian RMSD.
    # RMSD clustering across different receptor conformers is mathematically
    # incoherent — coordinate frames differ between conformers.
    # PLIF clustering compares pharmacophoric equivalence, which is the
    # scientifically correct and chemically meaningful similarity metric here.

    Parameters
    ----------
    docking_results     : list of DockingResult from dock_ensemble
    conformers          : list of ProDy AtomGroup (same order as docking_results)
    ligand_mol          : RDKit Mol of the docked ligand (for PLIF calculation)
    similarity_threshold: Tanimoto threshold; poses more similar than this are
                          merged into one cluster (distance = 1 - similarity)

    Returns
    -------
    ConsensusResult with ranked clusters.
    """
    all_poses: list[PoseResult] = []
    for dr in docking_results:
        all_poses.extend(dr.poses)

    if not all_poses:
        return _empty_result()

    # Compute PLIF fingerprints for every pose
    fps = _compute_plif_fingerprints(all_poses, conformers, ligand_mol)

    # Cluster by Tanimoto distance (1 - Tanimoto similarity)
    eps = 1.0 - similarity_threshold
    labels = _dbscan_cluster(fps, eps=eps)

    n_conformers_total = len(conformers) if conformers else max(
        (dr.conformer_index for dr in docking_results), default=0
    ) + 1

    clusters = _build_clusters(all_poses, fps, labels, n_conformers_total)

    if not clusters:
        return _empty_result()

    # Sort clusters: lowest (best) consensus score first
    clusters.sort(key=lambda c: c.consensus_score)
    top = clusters[0]

    pose_confidence = len(top.member_poses) / len(all_poses)
    site_confidence = _classify_site_confidence(top.score_std)

    return ConsensusResult(
        top_cluster=top,
        all_clusters=clusters,
        pose_confidence=pose_confidence,
        site_confidence=site_confidence,
        n_total_poses=len(all_poses),
        n_clusters=len(clusters),
    )


# ---------------------------------------------------------------------------
# PLIF fingerprinting
# ---------------------------------------------------------------------------

def _compute_plif_fingerprints(
    poses: list[PoseResult],
    conformers: list["prody.AtomGroup"],
    ligand_mol,
) -> np.ndarray:
    """
    Compute PLIF fingerprints for every pose using ProLIF.

    Receptors from UniDock/Vina are PDBQT (no hydrogens). pdb2pqr is used to
    add explicit H with PROPKA pKa-based protonation at pH 7.4 before passing
    to ProLIF, so H-bond donors/acceptors are correctly assigned.

    Protonation is cached per unique conformer — 5 conformers means 5 pdb2pqr
    calls (~15s total), not one per pose (which would be 450 calls).

    Falls back to coordinate pseudo-fingerprint if pdb2pqr or ProLIF fails.

    Returns float32 array of shape (n_poses, fp_length).
    """
    fps = []
    prolif_success = False

    if conformers and ligand_mol is not None:
        fps, prolif_success = _try_prolif(poses, conformers, ligand_mol)

    if not prolif_success:
        # Coordinate pseudo-fingerprint: bin atom distances to ligand centroid
        fps = _coordinate_pseudofp(poses)

    return np.array(fps, dtype=np.float32)


def _protonate_receptor(receptor_ag: "prody.AtomGroup", tmpdir: str) -> Optional[str]:
    """
    Write receptor to PDB, run pdb2pqr to add H with PROPKA pKa-based
    protonation at pH 7.4, return path to protonated PDB.
    Returns None if pdb2pqr is unavailable or fails.
    """
    import io, os, subprocess, shutil
    from prody import writePDBStream

    pdb2pqr_bin = shutil.which("pdb2pqr30") or shutil.which("pdb2pqr")
    if pdb2pqr_bin is None:
        return None

    raw_pdb = os.path.join(tmpdir, "raw.pdb")
    out_pdb = os.path.join(tmpdir, "protonated.pdb")
    out_pqr = os.path.join(tmpdir, "out.pqr")

    buf = io.StringIO()
    writePDBStream(buf, receptor_ag)
    with open(raw_pdb, "w") as fh:
        fh.write(buf.getvalue())

    try:
        result = subprocess.run(
            [pdb2pqr_bin, "--ff=AMBER", "--with-ph=7.4",
             "--pdb-output", out_pdb, raw_pdb, out_pqr],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and os.path.exists(out_pdb):
            return out_pdb
    except Exception:
        pass
    return None


def _try_prolif(
    poses: list[PoseResult],
    conformers: list["prody.AtomGroup"],
    ligand_mol,
) -> tuple[list[np.ndarray], bool]:
    """
    Compute ProLIF PLIF fingerprints for every pose.

    Receptors from UniDock are PDBQT (no H). pdb2pqr is used to add explicit
    hydrogens with PROPKA pH 7.4 protonation before ProLIF sees the receptor,
    so H-bond donors/acceptors are correctly assigned.

    Protonation is cached per conformer — 5 pdb2pqr calls total (not per pose).

    Returns (fps, success). On any failure returns ([], False).
    """
    try:
        import prolif
        import MDAnalysis as mda
        import io, os, shutil, tempfile
        from prody import writePDBStream

        _tmpdir = tempfile.mkdtemp()
        try:
            # --- Protonate each unique conformer once ---
            protonated_paths: dict[int, str] = {}
            for conf_idx, receptor_ag in enumerate(conformers):
                conf_tmpdir = os.path.join(_tmpdir, f"conf_{conf_idx}")
                os.makedirs(conf_tmpdir)
                prot_path = _protonate_receptor(receptor_ag, conf_tmpdir)
                if prot_path is None:
                    # pdb2pqr unavailable — write plain PDB
                    plain = os.path.join(conf_tmpdir, "plain.pdb")
                    buf = io.StringIO()
                    writePDBStream(buf, receptor_ag)
                    with open(plain, "w") as fh:
                        fh.write(buf.getvalue())
                    prot_path = plain
                protonated_paths[conf_idx] = prot_path

            # --- Compute fingerprint per pose ---
            fps = []
            for pose in poses:
                conf_idx = pose.conformer_index
                if conf_idx not in protonated_paths:
                    fps.append(np.zeros(128, dtype=np.float32))
                    continue

                lig_mol_pose = _pose_to_mol(ligand_mol, pose.coordinates)
                if lig_mol_pose is None:
                    fps.append(np.zeros(128, dtype=np.float32))
                    continue

                u_rec = mda.Universe(protonated_paths[conf_idx])
                prot = prolif.Molecule.from_mda(u_rec)
                lig = prolif.Molecule.from_rdkit(lig_mol_pose)
                fp = prolif.Fingerprint()
                fp.run_from_iterable([lig], prot)
                bv = fp.to_bitvectors()
                arr = np.array(list(bv[0]), dtype=np.float32) if bv else np.zeros(128, dtype=np.float32)
                fps.append(arr)

        finally:
            shutil.rmtree(_tmpdir, ignore_errors=True)

        if len(fps) == 0 or np.array(fps).sum() == 0:
            return [], False

        return fps, True

    except Exception as exc:
        warnings.warn(f"ProLIF failed ({type(exc).__name__}): {exc}. Falling back to pseudo-fingerprint.")
        return [], False


def _pose_to_mol(template_mol, coords: np.ndarray):
    """Set coordinates of template_mol to pose coords. Returns None on failure."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.RWMol(template_mol)
        # Remove Hs to match heavy-atom coord count
        mol = Chem.RemoveHs(mol)
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        conf = mol.GetConformer()
        n = min(mol.GetNumAtoms(), len(coords))
        for i in range(n):
            conf.SetAtomPosition(i, coords[i].tolist())
        return mol
    except Exception:
        return None


def _coordinate_pseudofp(poses: list[PoseResult]) -> list[np.ndarray]:
    """
    Fallback: represent each pose as a 64-bin histogram of its atom coordinates
    (normalised to unit norm). Dimensionally consistent across conformers since
    it captures pharmacophoric distribution rather than absolute position.
    """
    fps = []
    for pose in poses:
        coords = pose.coordinates
        if coords.size == 0:
            fps.append(np.zeros(64, dtype=np.float32))
            continue
        # Use pairwise distance distribution as fingerprint
        centroid = coords.mean(axis=0)
        dists = np.linalg.norm(coords - centroid, axis=1)
        hist, _ = np.histogram(dists, bins=64, range=(0.0, 20.0))
        arr = hist.astype(np.float32)
        norm = arr.sum()
        if norm > 0:
            arr /= norm
        fps.append(arr)
    return fps


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _dbscan_cluster(fps: np.ndarray, eps: float) -> np.ndarray:
    """
    DBSCAN clustering on PLIF fingerprint bit-vectors using Tanimoto distance.
    Returns integer label array; -1 = noise.
    """
    if len(fps) == 0:
        return np.array([], dtype=int)

    if len(fps) == 1:
        return np.array([0], dtype=int)

    # Tanimoto distance matrix
    dist_matrix = _tanimoto_distance_matrix(fps)

    db = DBSCAN(eps=eps, min_samples=1, metric="precomputed")
    labels = db.fit_predict(dist_matrix)
    return labels


def _tanimoto_distance_matrix(fps: np.ndarray) -> np.ndarray:
    """Compute pairwise Tanimoto distance (1 - Tanimoto) for binary/real fps."""
    n = len(fps)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            t = _tanimoto(fps[i], fps[j])
            d = 1.0 - t
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto coefficient for real-valued vectors."""
    dot = float(np.dot(a, b))
    denom = float(np.dot(a, a) + np.dot(b, b) - dot)
    if denom < 1e-9:
        return 1.0 if np.allclose(a, b) else 0.0
    return dot / denom


# ---------------------------------------------------------------------------
# Cluster assembly
# ---------------------------------------------------------------------------

def _build_clusters(
    poses: list[PoseResult],
    fps: np.ndarray,
    labels: np.ndarray,
    n_conformers_total: int,
) -> list[PoseCluster]:
    """Build PoseCluster objects from DBSCAN output."""
    unique_labels = set(labels)
    clusters = []

    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        members = [poses[i] for i in indices]
        member_fps = fps[indices]

        scores = np.array([p.score_kcal_mol for p in members])
        mean_score = float(scores.mean())
        score_std = float(scores.std()) if len(scores) > 1 else 0.0

        # Representative: best (lowest) scoring pose
        best_idx = int(np.argmin(scores))
        rep = members[best_idx]

        conf_set = {p.conformer_index for p in members}
        n_conf_rep = len(conf_set)
        fraction = n_conf_rep / max(n_conformers_total, 1)

        mean_fp = member_fps.mean(axis=0)

        # Consensus score: best pose score weighted by ensemble coverage.
        # Using the representative (best) pose score rather than mean_score
        # preserves discrimination between binders and non-binders — mean over
        # all 9 poses compresses everything toward -5/-6 and loses the signal.
        # Multiplying by fraction penalises clusters seen in few conformers.
        best_score = float(rep.score_kcal_mol)
        consensus_score = best_score * fraction

        clusters.append(PoseCluster(
            cluster_id=label,
            representative_pose=rep,
            member_poses=members,
            n_conformers_represented=n_conf_rep,
            fraction_ensemble=fraction,
            mean_score=mean_score,
            score_std=score_std,
            consensus_score=consensus_score,
            interaction_fingerprint=mean_fp,
        ))

    return clusters


def _classify_site_confidence(score_std: float) -> str:
    """Classify binding site confidence from score standard deviation."""
    if score_std < 1.0:
        return "high"
    elif score_std < 2.5:
        return "medium"
    else:
        return "low"


def _empty_result() -> ConsensusResult:
    dummy_pose = PoseResult(
        pose_index=0,
        score_kcal_mol=0.0,
        coordinates=np.zeros((1, 3), dtype=np.float32),
        conformer_index=0,
    )
    dummy_bb_import = None
    try:
        from spade.core.docking import BoundingBox
        dummy_bb = BoundingBox(center=np.zeros(3), size=np.full(3, 20.0))
    except Exception:
        dummy_bb = None  # type: ignore

    dummy_cluster = PoseCluster(
        cluster_id=0,
        representative_pose=dummy_pose,
        member_poses=[dummy_pose],
        n_conformers_represented=0,
        fraction_ensemble=0.0,
        mean_score=0.0,
        score_std=0.0,
        consensus_score=0.0,
        interaction_fingerprint=np.zeros(64, dtype=np.float32),
    )
    return ConsensusResult(
        top_cluster=dummy_cluster,
        all_clusters=[dummy_cluster],
        pose_confidence=0.0,
        site_confidence="low",
        n_total_poses=0,
        n_clusters=0,
    )
