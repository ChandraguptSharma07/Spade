"""
docking.py — Ensemble docking with pluggable CPU/GPU backends.

Backend options
---------------
"cpu"  — AutoDock Vina Python bindings (conda install -c conda-forge vina)
"gpu"  — AutoDock-GPU via subprocess   (conda install -c conda-forge autodock-gpu autogrid)

CRITICAL: The bounding box must be recomputed for EVERY conformer individually.
NMA perturbation shifts the geometric centre of the pocket by 1-2 Å across
conformers. A static bounding box means Vina docks off-centre in perturbed
conformers and returns scores for the wrong region — silently, without error.
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
# CPU backend — AutoDock Vina
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
# GPU backend — AutoDock-GPU
# ---------------------------------------------------------------------------

# AutoDock-GPU ships several binaries named by work-group size.
# We try them in order of preference (128 is the standard Kaggle/T4 choice).
_AUTODOCK_GPU_BINS = [
    "autodock_gpu_128wi",
    "autodock_gpu_64wi",
    "autodock_gpu_256wi",
    "autodock_gpu",
]

_AUTOGRID_BIN = "autogrid4"
_AUTODOCK_GPU_SPACING = 0.375   # Å — standard grid spacing


class AutoDockGPUDockingEngine(BaseDockingEngine):
    """
    GPU-accelerated backend using AutoDock-GPU + autogrid4.

    Install: conda install -c conda-forge autodock-gpu autogrid

    Scoring function: AutoDock4 (slightly different from Vina — scores are
    typically 0.5–1 kcal/mol more negative; not directly comparable to Vina
    results but internally consistent across runs).

    Parameters
    ----------
    n_runs       : number of independent GA runs per docking call (default 20).
                   More runs → better pose sampling, proportionally longer runtime.
    device_id    : GPU device index (default 0). Set to 1 on multi-GPU Kaggle nodes.
    """

    def __init__(self, n_runs: int = 20, device_id: int = 0) -> None:
        self._gpu_bin = self._find_binary(_AUTODOCK_GPU_BINS, "AutoDock-GPU")
        self._grid_bin = self._find_binary([_AUTOGRID_BIN], "autogrid4")
        self.n_runs = n_runs
        self.device_id = device_id

    @staticmethod
    def _find_binary(candidates: list[str], label: str) -> str:
        for name in candidates:
            path = shutil.which(name)
            if path:
                return path
        raise FileNotFoundError(
            f"{label} executable not found. "
            "Install with: conda install -c conda-forge autodock-gpu autogrid"
        )

    def dock(
        self,
        conformer: "prody.AtomGroup",
        ligand: PreparedLigand,
        bbox: BoundingBox,
        n_poses: int,
        conf_idx: int,
    ) -> list[PoseResult]:
        with tempfile.TemporaryDirectory() as tmpdir:
            return self._dock_in_tmpdir(conformer, ligand, bbox, n_poses, conf_idx, tmpdir)

    def _dock_in_tmpdir(
        self,
        conformer: "prody.AtomGroup",
        ligand: PreparedLigand,
        bbox: BoundingBox,
        n_poses: int,
        conf_idx: int,
        tmpdir: str,
    ) -> list[PoseResult]:
        receptor_pdbqt = os.path.join(tmpdir, "receptor.pdbqt")
        ligand_pdbqt = os.path.join(tmpdir, "ligand.pdbqt")
        gpf_path = os.path.join(tmpdir, "receptor.gpf")
        dpf_path = os.path.join(tmpdir, "ligand.dpf")
        dlg_path = os.path.join(tmpdir, "ligand.dlg")

        # Write receptor and ligand PDBQT files
        with open(receptor_pdbqt, "w") as fh:
            fh.write(_atomgroup_to_pdbqt(conformer))
        with open(ligand_pdbqt, "w") as fh:
            fh.write(ligand.pdbqt_string)

        # Determine atom types present in receptor and ligand
        receptor_types = _extract_atom_types(_atomgroup_to_pdbqt(conformer))
        ligand_types = _extract_atom_types(ligand.pdbqt_string)

        # Write GPF for autogrid4
        npts = _box_size_to_npts(bbox.size, _AUTODOCK_GPU_SPACING)
        gpf_content = _write_gpf(
            receptor_pdbqt=receptor_pdbqt,
            center=bbox.center,
            npts=npts,
            spacing=_AUTODOCK_GPU_SPACING,
            receptor_types=receptor_types,
            ligand_types=ligand_types,
        )
        with open(gpf_path, "w") as fh:
            fh.write(gpf_content)

        # Run autogrid4
        _run_subprocess(
            [self._grid_bin, "-p", gpf_path, "-l", os.path.join(tmpdir, "autogrid.log")],
            cwd=tmpdir,
            label="autogrid4",
        )

        # Write DPF for autodock_gpu
        fld_path = os.path.join(tmpdir, "receptor.maps.fld")
        dpf_content = _write_dpf(
            fld_path=fld_path,
            ligand_pdbqt=ligand_pdbqt,
            ligand_types=ligand_types,
            n_runs=self.n_runs,
        )
        with open(dpf_path, "w") as fh:
            fh.write(dpf_content)

        # Run autodock_gpu
        _run_subprocess(
            [
                self._gpu_bin,
                "--ffile", fld_path,
                "--lfile", ligand_pdbqt,
                "--resnam", os.path.join(tmpdir, "ligand"),
                "--nrun", str(self.n_runs),
                "--devnum", str(self.device_id + 1),  # autodock_gpu uses 1-based device index
            ],
            cwd=tmpdir,
            label="autodock_gpu",
        )

        return _parse_dlg(dlg_path, conf_idx, n_poses)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_docking_engine(
    backend: str = "cpu",
    exhaustiveness: int = 8,
    n_runs: int = 20,
    device_id: int = 0,
) -> BaseDockingEngine:
    """
    Factory function — return a configured docking engine.

    Parameters
    ----------
    backend         : "cpu" for Vina, "gpu" for AutoDock-GPU
    exhaustiveness  : (cpu only) Vina exhaustiveness parameter
    n_runs          : (gpu only) number of AutoDock-GPU GA runs
    device_id       : (gpu only) CUDA device index (0-based)
    """
    if backend == "cpu":
        return VinaDockingEngine(exhaustiveness=exhaustiveness)
    elif backend == "gpu":
        return AutoDockGPUDockingEngine(n_runs=n_runs, device_id=device_id)
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
    On Kaggle T4 x2, pass device_ids=[0, 1].

    Parameters
    ----------
    conformers      : list of ProDy AtomGroup conformers (from EnsembleGenerator)
    ligands         : list of PreparedLigand (may include multiple stereoisomers)
    pocket_residues : 0-based residue indices defining the pocket
    exhaustiveness  : (cpu) Vina exhaustiveness; ignored for gpu backend
    n_poses         : number of poses to return per ligand per conformer
    backend         : "cpu" (Vina) or "gpu" (AutoDock-GPU)
    device_ids      : (gpu only) list of CUDA device indices to use, e.g. [0, 1].
                      Defaults to [0] (single GPU). CPU backend ignores this.
    """
    if device_ids is None:
        device_ids = _detect_gpu_device_ids() if backend == "gpu" else [0]

    # Build the flat list of (conformer, ligand) jobs with per-job metadata
    jobs: list[tuple[int, "prody.AtomGroup", float, PreparedLigand, BoundingBox]] = []
    for conf_idx, conformer in enumerate(conformers):
        rmsd_data = conformer.getData("ca_rmsd_from_ref")
        ca_rmsd = float(rmsd_data[0]) if rmsd_data is not None else 0.0
        bbox = compute_bounding_box(conformer, pocket_residues)
        for ligand in ligands:
            jobs.append((conf_idx, conformer, ca_rmsd, ligand, bbox))

    n_workers = len(device_ids) if backend == "gpu" else 1
    n_runs = exhaustiveness * 3

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
            n_runs=n_runs,
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

    results: list[DockingResult] = []

    if n_workers == 1:
        # Single worker — no thread overhead, preserves original ordering
        for job_idx, (conf_idx, conformer, ca_rmsd, ligand, bbox) in enumerate(jobs):
            results.append(_run_job(job_idx, conf_idx, conformer, ca_rmsd, ligand, bbox))
    else:
        # Multi-GPU: submit all jobs, collect in original order
        futures_ordered = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for job_idx, (conf_idx, conformer, ca_rmsd, ligand, bbox) in enumerate(jobs):
                future = executor.submit(
                    _run_job, job_idx, conf_idx, conformer, ca_rmsd, ligand, bbox
                )
                futures_ordered.append(future)
        results = [f.result() for f in futures_ordered]

    return results


# ---------------------------------------------------------------------------
# AutoDock-GPU file writers
# ---------------------------------------------------------------------------

def _box_size_to_npts(size: np.ndarray, spacing: float) -> np.ndarray:
    """Convert box size in Å to number of grid points (must be even, max 126 for AutoDock-GPU)."""
    npts = np.ceil(size / spacing).astype(int)
    # Round up to next even number
    npts = np.where(npts % 2 == 0, npts, npts + 1)
    # AutoDock-GPU hard cap
    npts = np.minimum(npts, 126)
    return npts


def _write_gpf(
    receptor_pdbqt: str,
    center: np.ndarray,
    npts: np.ndarray,
    spacing: float,
    receptor_types: list[str],
    ligand_types: list[str],
) -> str:
    """Generate an autogrid4 GPF file content."""
    all_types = sorted(set(receptor_types) | set(ligand_types))
    map_lines = "\n".join(f"map receptor.{t}.map" for t in all_types)
    return (
        f"npts {npts[0]} {npts[1]} {npts[2]}\n"
        f"gridfld receptor.maps.fld\n"
        f"spacing {spacing}\n"
        f"receptor_types {' '.join(receptor_types)}\n"
        f"ligand_types {' '.join(ligand_types)}\n"
        f"receptor {receptor_pdbqt}\n"
        f"gridcenter {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n"
        f"smooth 0.5\n"
        f"{map_lines}\n"
        f"elecmap receptor.e.map\n"
        f"dsolvmap receptor.d.map\n"
        f"dielectric -0.1465\n"
    )


def _write_dpf(
    fld_path: str,
    ligand_pdbqt: str,
    ligand_types: list[str],
    n_runs: int,
) -> str:
    """Generate an AutoDock-GPU DPF file content."""
    map_lines = "\n".join(f"map receptor.{t}.map" for t in ligand_types)
    return (
        f"autodock_parameter_version 4.2\n"
        f"outlev 0\n"
        f"ligand_types {' '.join(ligand_types)}\n"
        f"fld {fld_path}\n"
        f"{map_lines}\n"
        f"elecmap receptor.e.map\n"
        f"dsolvmap receptor.d.map\n"
        f"move {ligand_pdbqt}\n"
        f"ga_pop_size 150\n"
        f"ga_num_evals 2500000\n"
        f"ga_run {n_runs}\n"
        f"rmstol 2.0\n"
        f"analysis\n"
    )


# ---------------------------------------------------------------------------
# AutoDock-GPU DLG parser
# ---------------------------------------------------------------------------

def _parse_dlg(dlg_path: str, conf_idx: int, n_poses: int) -> list[PoseResult]:
    """
    Parse an AutoDock-GPU .dlg output file into PoseResult objects.

    Each run in the DLG produces a docked model block starting with
    'DOCKED: MODEL'. Scores appear as:
      DOCKED: USER    Estimated Free Energy of Binding    =  -7.50 kcal/mol
    """
    if not os.path.exists(dlg_path):
        return []

    with open(dlg_path) as fh:
        content = fh.read()

    results: list[PoseResult] = []
    blocks = content.split("DOCKED: MODEL")[1:]  # first split is before any model

    for i, block in enumerate(blocks[:n_poses]):
        score: Optional[float] = None
        coords: list[list[float]] = []

        for line in block.splitlines():
            stripped = line.removeprefix("DOCKED: ")
            if "Estimated Free Energy of Binding" in stripped:
                try:
                    score = float(stripped.split("=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            elif stripped.startswith(("ATOM", "HETATM")):
                try:
                    x = float(stripped[30:38])
                    y = float(stripped[38:46])
                    z = float(stripped[46:54])
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


# ---------------------------------------------------------------------------
# Shared helpers
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
            ids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
            if ids:
                return ids
    except Exception:
        pass
    return [0]


def _run_subprocess(cmd: list[str], cwd: str, label: str) -> None:
    """Run an external command, raising RuntimeError on failure."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )


def _extract_atom_types(pdbqt_string: str) -> list[str]:
    """Return unique AutoDock atom types listed in a PDBQT string (last column of ATOM lines)."""
    types: set[str] = set()
    for line in pdbqt_string.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            parts = line.split()
            if parts:
                types.add(parts[-1])
    # Always include HD (polar hydrogen) and e/d maps for electrostatics/desolvation
    types.update({"HD"})
    return sorted(types)


def _atomgroup_to_pdbqt(ag: "prody.AtomGroup") -> str:
    """
    Convert a ProDy AtomGroup to a minimal PDBQT string for Vina or AutoDock-GPU.
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
    """Extract heavy-atom coordinates from a PDBQT pose block (Vina output)."""
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
