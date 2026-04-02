"""
docking.py — Ensemble docking with pluggable CPU/GPU backends.

Backend options
---------------
"cpu"  — AutoDock Vina Python bindings (pip install vina)
"gpu"  — gnina via subprocess            (pip install gnina)

Scoring functions
-----------------
cpu  : Vina empirical scoring function
gpu  : gnina CNN scoring function (Vina search + CNN rescore)
       Scores are NOT numerically comparable to CPU Vina — both rigid and
       ensemble runs must use the same backend for valid comparison.
       gnina is more accurate on pose prediction (McNutt et al. 2021).

CRITICAL: The bounding box must be recomputed for EVERY conformer individually.
NMA perturbation shifts the geometric centre of the pocket by 1-2 Å across
conformers. A static bounding box means the docking engine scores the wrong
region — silently, without error.
compute_bounding_box() is therefore a standalone function that must be called
inside the per-conformer loop, never cached.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

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

_GNINA_BINS = ["gnina"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Bounding box (shared by both backends)
# ---------------------------------------------------------------------------

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
        ca_all = conformer.select("name CA")
        if ca_all is not None:
            pocket_ca_coords = list(ca_all.getCoords())
        else:
            pocket_ca_coords = [conformer.getCoords().mean(axis=0)]

    coords = np.array(pocket_ca_coords)
    center = coords.mean(axis=0)

    if len(coords) > 1:
        span = coords.max(axis=0) - coords.min(axis=0)
    else:
        span = np.zeros(3)
    size = span + 2 * padding
    size = np.maximum(size, np.array([15.0, 15.0, 15.0]))

    return BoundingBox(center=center, size=size)


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class BaseDockingEngine(ABC):
    @abstractmethod
    def dock(
        self,
        conformer: "prody.AtomGroup",
        ligand: PreparedLigand,
        bbox: BoundingBox,
        n_poses: int,
        conf_idx: int,
    ) -> list[PoseResult]:
        """Dock one ligand against one conformer. Returns a list of PoseResult."""


# ---------------------------------------------------------------------------
# CPU backend — AutoDock Vina (Python bindings)
# ---------------------------------------------------------------------------

class VinaDockingEngine(BaseDockingEngine):
    """
    CPU backend using AutoDock Vina Python bindings.

    Install: conda install -c conda-forge vina
    """

    def __init__(self, exhaustiveness: int = 8) -> None:
        if not _VINA_AVAILABLE:
            raise ImportError(
                "AutoDock Vina Python bindings are required: "
                "conda install -c conda-forge vina"
            )
        self.exhaustiveness = exhaustiveness

    def dock(
        self,
        conformer: "prody.AtomGroup",
        ligand: PreparedLigand,
        bbox: BoundingBox,
        n_poses: int,
        conf_idx: int,
    ) -> list[PoseResult]:
        with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False, mode="w") as tmp:
            tmp.write(_atomgroup_to_pdbqt(conformer))
            receptor_path = tmp.name

        try:
            v = Vina(sf_name="vina", verbosity=0)
            v.set_receptor(receptor_path)
            v.set_ligand_from_string(ligand.pdbqt_string)
            v.compute_vina_maps(
                center=bbox.center.tolist(),
                box_size=bbox.size.tolist(),
            )
            v.dock(exhaustiveness=self.exhaustiveness, n_poses=n_poses)
            poses_pdbqt = v.poses(n_poses=n_poses)
            energies = v.energies(n_poses=n_poses)
        finally:
            os.unlink(receptor_path)

        results = []
        for i, (pdbqt_str, energy_row) in enumerate(
            zip(poses_pdbqt.split("MODEL")[1:], energies)
        ):
            score = float(energy_row[0])
            coords = _parse_pdbqt_coords(pdbqt_str)
            results.append(PoseResult(
                pose_index=i,
                score_kcal_mol=score,
                coordinates=coords,
                conformer_index=conf_idx,
            ))
        return results


# ---------------------------------------------------------------------------
# GPU backend — gnina
# ---------------------------------------------------------------------------

class GninaDockingEngine(BaseDockingEngine):
    """
    GPU-accelerated backend using gnina (Ragoza et al. 2017, McNutt et al. 2021).

    gnina uses Vina's Monte Carlo search with a CNN scoring function trained on
    PDBbind. More accurate than empirical Vina on pose prediction. Scores are
    NOT numerically comparable to CPU Vina — use the same backend for both
    rigid and ensemble runs when comparing.

    Install: pip install gnina

    Reference: McNutt et al. (2021) gnina 1.0: molecular docking with deep
    learning. J. Cheminformatics 13, 43. https://doi.org/10.1186/s13321-021-00522-2

    Parameters
    ----------
    exhaustiveness : Vina-style exhaustiveness (default 8)
    device_id      : CUDA device index, 0-based (default 0)
    cnn_scoring    : gnina CNN scoring mode — "rescore" (fast, recommended),
                     "refinement", or "metrorescore" (most accurate, slowest)
    """

    def __init__(
        self,
        exhaustiveness: int = 8,
        device_id: int = 0,
        cnn_scoring: str = "rescore",
    ) -> None:
        self._bin = _find_binary(_GNINA_BINS, "gnina")
        self.exhaustiveness = exhaustiveness
        self.device_id = device_id
        self.cnn_scoring = cnn_scoring

    def dock(
        self,
        conformer: "prody.AtomGroup",
        ligand: PreparedLigand,
        bbox: BoundingBox,
        n_poses: int,
        conf_idx: int,
    ) -> list[PoseResult]:
        with tempfile.TemporaryDirectory() as tmpdir:
            receptor_path = os.path.join(tmpdir, "receptor.pdbqt")
            ligand_path = os.path.join(tmpdir, "ligand.pdbqt")
            out_path = os.path.join(tmpdir, "out.sdf")  # gnina outputs SDF by default

            with open(receptor_path, "w") as fh:
                fh.write(_atomgroup_to_pdbqt(conformer))
            with open(ligand_path, "w") as fh:
                fh.write(ligand.pdbqt_string)

            cx, cy, cz = bbox.center
            sx, sy, sz = bbox.size

            _run_subprocess(
                [
                    self._bin,
                    "--receptor", receptor_path,
                    "--ligand", ligand_path,
                    "--center_x", f"{cx:.3f}",
                    "--center_y", f"{cy:.3f}",
                    "--center_z", f"{cz:.3f}",
                    "--size_x", f"{sx:.3f}",
                    "--size_y", f"{sy:.3f}",
                    "--size_z", f"{sz:.3f}",
                    "--exhaustiveness", str(self.exhaustiveness),
                    "--num_modes", str(n_poses),
                    "--device", str(self.device_id),
                    "--cnn_scoring", self.cnn_scoring,
                    "--out", out_path,
                    "--quiet",
                ],
                cwd=tmpdir,
                label="gnina",
            )

            return _parse_gnina_sdf_output(out_path, conf_idx, n_poses)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _find_binary(candidates: list[str], label: str) -> str:
    """
    Locate a binary by name, checking PATH and common Kaggle/Colab install paths.
    Raises FileNotFoundError with install instructions if not found.
    """
    search_prefixes = ["", "/opt/conda/bin/", "/usr/local/bin/", "/usr/bin/"]
    for name in candidates:
        for prefix in search_prefixes:
            path = shutil.which(name) if not prefix else None
            candidate_path = f"{prefix}{name}" if prefix else None
            if path:
                return path
            if candidate_path and os.path.isfile(candidate_path) and os.path.getsize(candidate_path) > 1000:
                return candidate_path
    raise FileNotFoundError(
        f"{label} executable not found. Install with: pip install {candidates[0]}"
    )


def get_docking_engine(
    backend: str = "cpu",
    exhaustiveness: int = 8,
    device_id: int = 0,
    cnn_scoring: str = "rescore",
) -> BaseDockingEngine:
    """
    Factory — return a configured docking engine.

    Parameters
    ----------
    backend        : "cpu" (Vina) or "gpu" (gnina)
    exhaustiveness : search exhaustiveness, same meaning for both backends
    device_id      : (gpu only) CUDA device index, 0-based
    cnn_scoring    : (gpu only) gnina CNN scoring mode: "rescore", "refinement",
                     or "metrorescore"
    """
    if backend == "cpu":
        return VinaDockingEngine(exhaustiveness=exhaustiveness)
    elif backend == "gpu":
        return GninaDockingEngine(
            exhaustiveness=exhaustiveness,
            device_id=device_id,
            cnn_scoring=cnn_scoring,
        )
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'cpu' or 'gpu'.")


def dock_ensemble(
    conformers: list["prody.AtomGroup"],
    ligands: list[PreparedLigand],
    pocket_residues: np.ndarray,
    exhaustiveness: int = 8,
    n_poses: int = 9,
    backend: str = "cpu",
    device_ids: Optional[list[int]] = None,
) -> list[DockingResult]:
    """
    Dock all ligands against every conformer in the ensemble.

    For the GPU backend, multiple device_ids enables multi-GPU parallelism —
    jobs are distributed across devices using a thread pool (one thread per GPU).
    Kaggle T4 x2 is handled automatically when device_ids is not specified.

    Parameters
    ----------
    conformers      : list of ProDy AtomGroup conformers (from EnsembleGenerator)
    ligands         : list of PreparedLigand (may include multiple stereoisomers)
    pocket_residues : 0-based residue indices defining the pocket
    exhaustiveness  : Vina exhaustiveness; same meaning for cpu and gpu backends
    n_poses         : number of poses to return per ligand per conformer
    backend         : "cpu" (Vina) or "gpu" (gnina)
    device_ids      : (gpu only) list of CUDA device indices to use.
                      Defaults to None — auto-detected via nvidia-smi.
                      CPU backend ignores this parameter.
    """
    if device_ids is None:
        device_ids = _detect_gpu_device_ids() if backend == "gpu" else [0]

    # Build flat job list with per-job metadata
    jobs: list[tuple[int, "prody.AtomGroup", float, PreparedLigand, BoundingBox]] = []
    for conf_idx, conformer in enumerate(conformers):
        rmsd_data = conformer.getData("ca_rmsd_from_ref")
        ca_rmsd = float(rmsd_data[0]) if rmsd_data is not None else 0.0
        bbox = compute_bounding_box(conformer, pocket_residues)
        for ligand in ligands:
            jobs.append((conf_idx, conformer, ca_rmsd, ligand, bbox))

    n_workers = len(device_ids) if backend == "gpu" else 1

    def _run_job(
        job_idx: int,
        conf_idx: int,
        conformer: "prody.AtomGroup",
        ca_rmsd: float,
        ligand: PreparedLigand,
        bbox: BoundingBox,
    ) -> DockingResult:
        device_id = device_ids[job_idx % len(device_ids)]
        engine = get_docking_engine(
            backend=backend,
            exhaustiveness=exhaustiveness,
            device_id=device_id,
        )
        t0 = time.perf_counter()
        poses = engine.dock(conformer, ligand, bbox, n_poses, conf_idx)
        elapsed = time.perf_counter() - t0
        return DockingResult(
            conformer_index=conf_idx,
            conformer_ca_rmsd=ca_rmsd,
            poses=poses,
            bounding_box=bbox,
            docking_time_seconds=elapsed,
        )

    if n_workers == 1:
        return [
            _run_job(job_idx, *job)
            for job_idx, job in enumerate(jobs)
        ]

    # Multi-GPU: submit all jobs, preserve original ordering
    futures_ordered = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for job_idx, job in enumerate(jobs):
            futures_ordered.append(executor.submit(_run_job, job_idx, *job))
    return [f.result() for f in futures_ordered]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_gpu_device_ids() -> list[int]:
    """
    Return a list of available CUDA device indices by querying nvidia-smi.
    Falls back to [0] if nvidia-smi is unavailable or reports no devices.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return [0]
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            ids = [
                int(line.strip())
                for line in result.stdout.splitlines()
                if line.strip().isdigit()
            ]
            if ids:
                return ids
    except Exception:
        pass
    return [0]


def _run_subprocess(cmd: list[str], cwd: str, label: str) -> None:
    """Run an external command, raising RuntimeError with output on failure."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )


def _parse_vina_pdbqt_output(
    out_path: str,
    conf_idx: int,
    n_poses: int,
) -> list[PoseResult]:
    """
    Parse a Vina-GPU output PDBQT file into PoseResult objects.

    Vina (CPU and GPU) writes scores as:
      REMARK VINA RESULT:   -7.50      0.000      0.000
    followed by ATOM/HETATM lines for that pose.
    """
    if not os.path.exists(out_path):
        return []

    with open(out_path) as fh:
        content = fh.read()

    results: list[PoseResult] = []
    for i, block in enumerate(content.split("MODEL")[1:n_poses + 1]):
        score: Optional[float] = None
        coords: list[list[float]] = []

        for line in block.splitlines():
            if line.startswith("REMARK VINA RESULT:"):
                try:
                    score = float(line.split(":")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            elif line.startswith(("ATOM", "HETATM")):
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    coords.append([x, y, z])
                except (ValueError, IndexError):
                    continue

        if score is None:
            continue

        results.append(PoseResult(
            pose_index=i,
            score_kcal_mol=score,
            coordinates=np.array(coords, dtype=np.float32) if coords else np.zeros((1, 3), dtype=np.float32),
            conformer_index=conf_idx,
        ))

    return results


def _parse_gnina_sdf_output(
    out_path: str,
    conf_idx: int,
    n_poses: int,
) -> list[PoseResult]:
    """
    Parse a gnina output SDF file into PoseResult objects.

    gnina writes one molecule block per pose. Each block contains a property:
      > <minimizedAffinity>
      -7.50

    Coordinates are in the standard SDF atom block (columns 1-3 of the atom table).
    """
    if not os.path.exists(out_path):
        return []

    with open(out_path) as fh:
        content = fh.read()

    results: list[PoseResult] = []
    # SDF molecules are separated by "$$$$"
    blocks = [b.strip() for b in content.split("$$$$") if b.strip()]

    for i, block in enumerate(blocks[:n_poses]):
        score: Optional[float] = None
        coords: list[list[float]] = []
        lines = block.splitlines()

        # Parse score from SDF property block
        for j, line in enumerate(lines):
            if "minimizedAffinity" in line or "CNNaffinity" in line:
                # Value is on the next non-empty line
                for k in range(j + 1, min(j + 4, len(lines))):
                    try:
                        score = float(lines[k].strip())
                        break
                    except ValueError:
                        continue
                if score is not None:
                    break

        # Parse coordinates from atom block
        # SDF atom block starts after the counts line (4th line of header)
        # Format: xxxxx.xxxxyyyyy.yyyyzzzzz.zzzz ...
        counts_line_idx = 3  # 0-indexed: title, program, comment, counts
        if len(lines) > counts_line_idx:
            try:
                n_atoms = int(lines[counts_line_idx].split()[0])
                for atom_line in lines[counts_line_idx + 1: counts_line_idx + 1 + n_atoms]:
                    parts = atom_line.split()
                    if len(parts) >= 3:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        coords.append([x, y, z])
            except (ValueError, IndexError):
                pass

        if score is None:
            continue

        results.append(PoseResult(
            pose_index=i,
            score_kcal_mol=score,
            coordinates=np.array(coords, dtype=np.float32) if coords else np.zeros((1, 3), dtype=np.float32),
            conformer_index=conf_idx,
        ))

    return results


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
            continue
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
    """Extract heavy-atom coordinates from a PDBQT pose block (Vina CPU output)."""
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
