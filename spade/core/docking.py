"""
docking.py — Ensemble docking with pluggable CPU/GPU backends.

Backend options
---------------
"cpu"  — AutoDock Vina Python bindings   (conda install -c conda-forge vina)
"gpu"  — UniDock via subprocess           (conda install -c conda-forge unidock)

Scoring functions
-----------------
Both backends use the same Vina empirical scoring function by default.
Scores are numerically comparable across backends — rigid CPU and ensemble GPU
results can be placed on the same axis in plots.

UniDock scoring options: "vina" (default), "vinardo", "ad4".
Use the same scoring value for both rigid and ensemble runs.

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

_UNIDOCK_BINS = ["unidock"]


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
    # Conform to Vina constraints: Minimum 15A^3 for stability, Maximum 30A^3 to prevent UniDock memory segfaults
    size = np.clip(size, 15.0, 30.0)

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
# GPU backend — UniDock
# ---------------------------------------------------------------------------

class UniDockDockingEngine(BaseDockingEngine):
    """
    GPU-accelerated backend using UniDock (Yu et al. 2023, JCTC).

    UniDock parallelises the Vina Monte Carlo search across GPU CUDA cores —
    the search step itself runs on GPU, not just scoring. Scoring function is
    Vina-compatible, so scores are directly comparable to VinaDockingEngine.

    Install: conda install -c conda-forge unidock

    Reference: Yu et al. (2023) Uni-Dock: GPU-Accelerated Docking Enables
    Ultralarge Virtual Screening. J. Chem. Theory Comput. 19, 3336–3345.
    https://doi.org/10.1021/acs.jctc.2c01145

    Parameters
    ----------
    exhaustiveness : search exhaustiveness (default 8); ignored when
                     search_mode is set
    scoring        : scoring function — "vina" (default), "vinardo", or "ad4"
    device_id      : CUDA device index, 0-based (default 0)
    search_mode    : preset that overrides exhaustiveness — None (default),
                     "fast" (exhaustiveness=128, max_step=20),
                     "balance" (exhaustiveness=384, max_step=40),
                     "detail" (exhaustiveness=512, max_step=40)
    """

    def __init__(
        self,
        exhaustiveness: int = 8,
        scoring: str = "vina",
        device_id: int = 0,
        search_mode: Optional[str] = None,
    ) -> None:
        self._bin = _find_binary(_UNIDOCK_BINS, "unidock")
        self.exhaustiveness = exhaustiveness
        self.scoring = scoring
        self.device_id = device_id
        self.search_mode = search_mode

    def dock(
        self,
        conformer: "prody.AtomGroup",
        ligand: PreparedLigand,
        bbox: BoundingBox,
        n_poses: int,
        conf_idx: int,
        seed: Optional[int] = None,
    ) -> list[PoseResult]:
        """Dock a single ligand against one conformer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            receptor_path = os.path.join(tmpdir, "receptor.pdbqt")
            ligand_path = os.path.join(tmpdir, "ligand.pdbqt")
            out_path = os.path.join(tmpdir, "out.pdbqt")

            with open(receptor_path, "w") as fh:
                fh.write(_atomgroup_to_pdbqt(conformer))
            with open(ligand_path, "w") as fh:
                fh.write(ligand.pdbqt_string)

            cmd = self._build_cmd(receptor_path, bbox, n_poses, seed=seed)
            cmd += ["--ligand", ligand_path, "--out", out_path]

            _run_subprocess(cmd, cwd=tmpdir, label="unidock")
            return _parse_vina_pdbqt_output(out_path, conf_idx, n_poses)

    def dock_batch(
        self,
        conformer: "prody.AtomGroup",
        ligands: list[PreparedLigand],
        bbox: BoundingBox,
        n_poses: int,
        conf_idx: int,
        seed: Optional[int] = None,
    ) -> list[list[PoseResult]]:
        """
        Dock multiple ligands against one conformer in a single GPU call using
        UniDock's --gpu_batch flag. All ligands are submitted together so the
        GPU kernel stays occupied for the full batch duration, producing real
        GPU utilisation instead of a series of short-lived subprocesses.

        Falls back to sequential dock() calls if --gpu_batch fails (e.g. on
        unusual topologies or driver incompatibilities).
        """
        if not ligands:
            return []
        try:
            return self._dock_batch_gpu(conformer, ligands, bbox, n_poses, conf_idx, seed=seed)
        except Exception as _e:
            import warnings
            warnings.warn(
                f"UniDock --gpu_batch failed, falling back to sequential dock() calls.\n"
                f"Error: {_e}",
                stacklevel=2,
            )
            results = []
            for lig in ligands:
                results.append(self.dock(conformer, lig, bbox, n_poses, conf_idx, seed=seed))
            return results

    def _dock_batch_gpu(
        self,
        conformer: "prody.AtomGroup",
        ligands: list[PreparedLigand],
        bbox: BoundingBox,
        n_poses: int,
        conf_idx: int,
        seed: Optional[int] = None,
    ) -> list[list[PoseResult]]:
        """Internal: true --gpu_batch call — all ligands in one subprocess."""
        with tempfile.TemporaryDirectory() as tmpdir:
            receptor_path = os.path.join(tmpdir, "receptor.pdbqt")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(out_dir)

            with open(receptor_path, "w") as fh:
                fh.write(_atomgroup_to_pdbqt(conformer))

            ligand_paths = []
            for i, lig in enumerate(ligands):
                lpath = os.path.join(tmpdir, f"lig_{i:04d}.pdbqt")
                with open(lpath, "w") as fh:
                    fh.write(lig.pdbqt_string)
                ligand_paths.append(lpath)

            cmd = self._build_cmd(receptor_path, bbox, n_poses, seed=seed)
            # Replace --receptor (already in cmd) with gpu_batch args
            cmd += ["--gpu_batch"] + ligand_paths + ["--dir", out_dir]

            _run_subprocess(cmd, cwd=tmpdir, label="unidock-batch")

            results = []
            for i, lpath in enumerate(ligand_paths):
                stem = os.path.splitext(os.path.basename(lpath))[0]
                out_file = os.path.join(out_dir, f"{stem}_out.pdbqt")
                results.append(_parse_vina_pdbqt_output(out_file, conf_idx, n_poses))
            return results

    def _build_cmd(
        self,
        receptor_path: str,
        bbox: BoundingBox,
        n_poses: int,
        seed: Optional[int] = None,
    ) -> list[str]:
        cx, cy, cz = bbox.center
        sx, sy, sz = bbox.size
        cmd = [
            self._bin,
            "--receptor", receptor_path,
            "--center_x", f"{cx:.3f}",
            "--center_y", f"{cy:.3f}",
            "--center_z", f"{cz:.3f}",
            "--size_x", f"{sx:.3f}",
            "--size_y", f"{sy:.3f}",
            "--size_z", f"{sz:.3f}",
            "--num_modes", str(n_poses),
            "--scoring", self.scoring,
            "--device_id", str(self.device_id),
        ]
        if self.search_mode:
            cmd += ["--search_mode", self.search_mode]
        else:
            cmd += ["--exhaustiveness", str(self.exhaustiveness)]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        return cmd


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
    scoring: str = "vina",
    search_mode: Optional[str] = None,
) -> BaseDockingEngine:
    """
    Factory — return a configured docking engine.

    Parameters
    ----------
    backend        : "cpu" (Vina) or "gpu" (UniDock)
    exhaustiveness : search exhaustiveness; same meaning for both backends.
                     Ignored when search_mode is set (gpu only).
    device_id      : (gpu only) CUDA device index, 0-based
    scoring        : (gpu only) scoring function — "vina" (default),
                     "vinardo", or "ad4". Use "vina" to keep scores
                     comparable with the cpu backend.
    search_mode    : (gpu only) UniDock preset — None, "fast", "balance",
                     or "detail". Overrides exhaustiveness when set.
    """
    if backend == "cpu":
        return VinaDockingEngine(exhaustiveness=exhaustiveness)
    elif backend == "gpu":
        return UniDockDockingEngine(
            exhaustiveness=exhaustiveness,
            scoring=scoring,
            device_id=device_id,
            search_mode=search_mode,
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
    scoring: str = "vina",
    search_mode: Optional[str] = None,
    base_seed: Optional[int] = None,
) -> list[DockingResult]:
    """
    Dock all ligands against every conformer in the ensemble.

    For the GPU backend (UniDock), all ligands for each conformer are batched
    into a single GPU call — this is more efficient than one call per ligand.
    Multiple device_ids enables multi-GPU parallelism (one conformer per GPU at
    a time); Kaggle T4 x2 is handled automatically when device_ids is not set.

    Parameters
    ----------
    conformers      : list of ProDy AtomGroup conformers (from EnsembleGenerator)
    ligands         : list of PreparedLigand (may include multiple stereoisomers)
    pocket_residues : 0-based residue indices defining the pocket
    exhaustiveness  : Vina exhaustiveness; same meaning for cpu and gpu.
                      Ignored when search_mode is set.
    n_poses         : number of poses to return per ligand per conformer
    backend         : "cpu" (Vina) or "gpu" (UniDock)
    device_ids      : (gpu only) CUDA device indices to use; auto-detected
                      from nvidia-smi when None. CPU backend ignores this.
    scoring         : (gpu only) "vina" (default, comparable to cpu),
                      "vinardo", or "ad4"
    search_mode     : (gpu only) UniDock preset — "fast", "balance", or
                      "detail". Overrides exhaustiveness when set.
    """
    if device_ids is None:
        device_ids = _detect_gpu_device_ids() if backend == "gpu" else [0]

    n_workers = len(device_ids) if backend == "gpu" else 1

    # --- GPU path: batch all ligands per conformer for efficiency ---
    if backend == "gpu":
        def _run_conformer(args: tuple) -> list[DockingResult]:
            conf_idx, conformer, device_id = args
            rmsd_data = conformer.getData("ca_rmsd_from_ref")
            ca_rmsd = float(rmsd_data[0]) if rmsd_data is not None else 0.0
            bbox = compute_bounding_box(conformer, pocket_residues)
            engine = get_docking_engine(
                backend="gpu",
                exhaustiveness=exhaustiveness,
                device_id=device_id,
                scoring=scoring,
                search_mode=search_mode,
            )
            seed = (base_seed + conf_idx) if base_seed is not None else None
            t0 = time.perf_counter()
            batch_poses = engine.dock_batch(conformer, ligands, bbox, n_poses, conf_idx, seed=seed)
            elapsed = time.perf_counter() - t0
            per_ligand_time = elapsed / max(len(ligands), 1)
            return [
                DockingResult(
                    conformer_index=conf_idx,
                    conformer_ca_rmsd=ca_rmsd,
                    poses=poses,
                    bounding_box=bbox,
                    docking_time_seconds=per_ligand_time,
                )
                for poses in batch_poses
            ]

        conf_args = [
            (conf_idx, conformer, device_ids[conf_idx % len(device_ids)])
            for conf_idx, conformer in enumerate(conformers)
        ]

        if n_workers == 1:
            results = []
            n_conf = len(conf_args)
            for args in conf_args:
                conf_idx = args[0]
                print(f"  [GPU] conformer {conf_idx + 1}/{n_conf} ...", flush=True)
                results.extend(_run_conformer(args))
            return results

        futures_ordered = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for args in conf_args:
                futures_ordered.append(executor.submit(_run_conformer, args))
        results = []
        for f in futures_ordered:
            results.extend(f.result())
        return results

    # --- CPU path: flat job list, single-threaded ---
    results = []
    for conf_idx, conformer in enumerate(conformers):
        rmsd_data = conformer.getData("ca_rmsd_from_ref")
        ca_rmsd = float(rmsd_data[0]) if rmsd_data is not None else 0.0
        bbox = compute_bounding_box(conformer, pocket_residues)
        engine = get_docking_engine(backend="cpu", exhaustiveness=exhaustiveness)
        for ligand in ligands:
            seed = (base_seed + conf_idx) if base_seed is not None else None
            t0 = time.perf_counter()
            poses = engine.dock(conformer, ligand, bbox, n_poses, conf_idx, seed=seed)
            elapsed = time.perf_counter() - t0
            results.append(DockingResult(
                conformer_index=conf_idx,
                conformer_ca_rmsd=ca_rmsd,
                poses=poses,
                bounding_box=bbox,
                docking_time_seconds=elapsed,
            ))
    return results


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


def _run_subprocess(cmd: list[str], cwd: str, label: str) -> str:
    """Run an external command, raising RuntimeError with output on failure.
    Returns combined stdout+stderr on success for caller inspection."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    combined = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
    return combined


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
        warnings.warn(
            f"UniDock produced no output file at {out_path} (conformer {conf_idx}). "
            "This usually means CUDA failed to initialize — check that the UniDock binary "
            "matches your CUDA version, or set backend='cpu' to use Vina instead.",
            stacklevel=3,
        )
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
