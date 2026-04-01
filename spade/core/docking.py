"""
docking.py — Ensemble docking via Vina Python bindings.

CRITICAL: The bounding box must be recomputed for EVERY conformer individually.
NMA perturbation shifts the geometric centre of the pocket by 1-2 Å across
conformers. A static bounding box means Vina docks off-centre in perturbed
conformers and returns scores for the wrong region — silently, without error.
compute_bounding_box() is therefore a standalone function that must be called
inside the per-conformer loop, never cached.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

try:
    import prody
    prody.confProDy(verbosity="none")
except ImportError as e:
    raise ImportError("ProDy is required: conda install -c conda-forge prody") from e

from spade.core.ligand import PreparedLigand

_VINA_AVAILABLE = False
try:
    from vina import Vina
    _VINA_AVAILABLE = True
except ImportError:
    pass


@dataclass
class BoundingBox:
    """
    Docking bounding box centred on the pocket Cα centroid of a specific conformer.
    Must be recomputed per conformer — never reused across conformers.
    """
    center: np.ndarray   # (3,) — x, y, z in Angstrom
    size: np.ndarray     # (3,) — box dimensions in Angstrom


@dataclass
class PoseResult:
    """A single docked pose."""
    pose_index: int
    score_kcal_mol: float
    coordinates: np.ndarray    # (n_heavy_atoms, 3)
    conformer_index: int


@dataclass
class DockingResult:
    """All poses from docking one ligand against one conformer."""
    conformer_index: int
    conformer_ca_rmsd: float
    poses: list[PoseResult]
    bounding_box: BoundingBox  # the box used for THIS conformer — stored for provenance
    docking_time_seconds: float


def compute_bounding_box(
    conformer: "prody.AtomGroup",
    pocket_residues: np.ndarray,
    padding: float = 10.0,
) -> BoundingBox:
    """
    Compute a bounding box centred on the Cα centroid of pocket_residues
    in the given conformer.

    This function MUST be called per-conformer. The pocket centroid shifts
    with NMA displacement — a static box computed from the reference structure
    will be off-centre for perturbed conformers, causing Vina to score the
    wrong region without producing any error or warning.

    Parameters
    ----------
    conformer       : ProDy AtomGroup for this specific conformer
    pocket_residues : 0-based residue indices defining the pocket
    padding         : Angstrom added to each side of the min/max bounding box

    Returns
    -------
    BoundingBox with center and size derived from this conformer's coordinates.
    """
    residues = list(conformer.iterResidues())
    pocket_ca_coords = []

    for idx in pocket_residues:
        if idx >= len(residues):
            continue
        res = residues[idx]
        ca = res.select("name CA")
        if ca is not None and len(ca) > 0:
            pocket_ca_coords.append(ca.getCoords()[0])

    if not pocket_ca_coords:
        # Fallback: centre on all Cα atoms
        ca_all = conformer.select("name CA")
        if ca_all is not None:
            pocket_ca_coords = list(ca_all.getCoords())
        else:
            pocket_ca_coords = [conformer.getCoords().mean(axis=0)]

    coords = np.array(pocket_ca_coords)
    center = coords.mean(axis=0)

    # Box size: span of pocket Cα coords + padding on all sides
    if len(coords) > 1:
        span = coords.max(axis=0) - coords.min(axis=0)
    else:
        span = np.zeros(3)
    size = span + 2 * padding

    # Minimum box size to avoid Vina errors
    size = np.maximum(size, np.array([15.0, 15.0, 15.0]))

    return BoundingBox(center=center, size=size)


def dock_ensemble(
    conformers: list["prody.AtomGroup"],
    ligands: list[PreparedLigand],
    pocket_residues: np.ndarray,
    exhaustiveness: int = 8,
    n_poses: int = 9,
) -> list[DockingResult]:
    """
    Dock all ligands against every conformer in the ensemble.

    For each conformer:
      1. Compute bounding box from THIS conformer's pocket Cα centroid
      2. Initialise Vina with this conformer's coordinates
      3. Dock all prepared ligands
      4. Collect all poses with full provenance

    The bounding box is NEVER reused across conformers.

    Parameters
    ----------
    conformers      : list of ProDy AtomGroup conformers (from EnsembleGenerator)
    ligands         : list of PreparedLigand (may include multiple stereoisomers)
    pocket_residues : 0-based residue indices defining the pocket
    exhaustiveness  : Vina exhaustiveness parameter
    n_poses         : number of poses to return per ligand per conformer

    Returns
    -------
    list[DockingResult], one per (conformer, ligand) pair
    """
    if not _VINA_AVAILABLE:
        raise ImportError(
            "AutoDock Vina Python bindings are required: "
            "conda install -c conda-forge vina"
        )

    results: list[DockingResult] = []

    for conf_idx, conformer in enumerate(conformers):
        # Retrieve provenance RMSD stored by EnsembleGenerator
        rmsd_data = conformer.getData("ca_rmsd_from_ref")
        ca_rmsd = float(rmsd_data[0]) if rmsd_data is not None else 0.0

        # Per-conformer bounding box — critical, must not be cached
        bbox = compute_bounding_box(conformer, pocket_residues)

        for ligand in ligands:
            t0 = time.perf_counter()
            poses = _dock_single(
                conformer, ligand, bbox, exhaustiveness, n_poses, conf_idx
            )
            elapsed = time.perf_counter() - t0

            results.append(DockingResult(
                conformer_index=conf_idx,
                conformer_ca_rmsd=ca_rmsd,
                poses=poses,
                bounding_box=bbox,
                docking_time_seconds=elapsed,
            ))

    return results


def _dock_single(
    conformer: "prody.AtomGroup",
    ligand: PreparedLigand,
    bbox: BoundingBox,
    exhaustiveness: int,
    n_poses: int,
    conf_idx: int,
) -> list[PoseResult]:
    """Run Vina for one (conformer, ligand) pair and return pose results."""
    import tempfile, os

    v = Vina(sf_name="vina", verbosity=0)

    # Write receptor to a temporary PDB file and load into Vina
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False, mode="w") as tmp:
        tmp.write(_atomgroup_to_pdbqt(conformer))
        receptor_path = tmp.name

    try:
        v.set_receptor(receptor_path)
        v.set_ligand_from_string(ligand.pdbqt_string)
        v.compute_vina_maps(
            center=bbox.center.tolist(),
            box_size=bbox.size.tolist(),
        )
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        poses_pdbqt = v.poses(n_poses=n_poses)
        energies = v.energies(n_poses=n_poses)
    finally:
        os.unlink(receptor_path)

    pose_results = []
    for i, (pdbqt_str, energy_row) in enumerate(zip(poses_pdbqt.split("MODEL")[1:], energies)):
        score = float(energy_row[0])
        coords = _parse_pdbqt_coords(pdbqt_str)
        pose_results.append(PoseResult(
            pose_index=i,
            score_kcal_mol=score,
            coordinates=coords,
            conformer_index=conf_idx,
        ))

    return pose_results


def _atomgroup_to_pdbqt(ag: "prody.AtomGroup") -> str:
    """
    Convert a ProDy AtomGroup to a minimal PDBQT string for Vina.
    Uses only heavy atoms; assigns AutoDock atom types naively.
    For production use, prepare the receptor with prepare_receptor4.py or Meeko.
    """
    lines = []
    for atom in ag.iterAtoms():
        element = (atom.getElement() or atom.getName()[0]).upper()
        if element == "H":
            continue  # skip hydrogens
        ad_type = _ELEMENT_TO_AD_TYPE.get(element, "C")
        x, y, z = atom.getCoords()
        resname = atom.getResname()
        resnum = atom.getResnum()
        chain = atom.getChid() or "A"
        name = atom.getName().ljust(4)
        lines.append(
            f"ATOM  {atom.getSerial():5d} {name} {resname:3s} {chain}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    {0.000:6.3f} {ad_type}"
        )
    lines.append("TER\n")
    return "\n".join(lines)


def _parse_pdbqt_coords(pdbqt_block: str) -> np.ndarray:
    """Extract heavy-atom coordinates from a PDBQT pose block."""
    coords = []
    for line in pdbqt_block.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords.append([x, y, z])
            except (ValueError, IndexError):
                continue
    return np.array(coords, dtype=np.float32) if coords else np.zeros((1, 3), dtype=np.float32)


# Minimal AutoDock atom type mapping
_ELEMENT_TO_AD_TYPE: dict[str, str] = {
    "C": "C", "N": "NA", "O": "OA", "S": "SA",
    "P": "P", "F": "F", "CL": "CL", "BR": "BR",
    "I": "I", "ZN": "Zn", "MG": "Mg", "CA": "Ca",
    "FE": "Fe", "MN": "Mn",
}
