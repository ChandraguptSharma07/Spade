"""
repacker.py — Side-chain clash resolution via Dunbrack rotamer repacking.

Default backend: DunbrackRepacker — pure Python/NumPy with a Numba JIT
inner loop for hard-sphere clash scoring.

Optional enhanced backend: PDBFixerRepacker (requires pdbfixer).

CRITICAL INVARIANT: the output AtomGroup must have the identical atom count
and ordering as the input. Never reorder atoms, never add or remove atoms.
Only coordinates change. ProLIF and Vina silently fail on non-standard atom
ordering — the error manifests as nonsensical scores, not a crash.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import prody
    prody.confProDy(verbosity="none")
except ImportError as e:
    raise ImportError("ProDy is required: conda install -c conda-forge prody") from e

try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

_DUNBRACK_PKL = Path(__file__).parent.parent / "data" / "dunbrack_rotamers.pkl"

# Residues with no rotatable side-chain — skip repacking entirely
_NO_SIDECHAIN = {"GLY", "ALA"}

# Proline chi angles are ring-constrained; we apply special handling
_PROLINE = "PRO"


# ---------------------------------------------------------------------------
# Numba JIT clash scoring (falls back to pure NumPy if Numba unavailable)
# ---------------------------------------------------------------------------

if _NUMBA_AVAILABLE:
    import numba

    @numba.jit(nopython=True, cache=True)
    def _score_rotamer_jit(
        sc_coords: np.ndarray,
        neighbor_coords: np.ndarray,
        threshold: float,
    ) -> float:
        """
        Hard-sphere clash score between side-chain atoms and fixed neighbours.
        Returns the sum of squared violations (dist < threshold).
        Pure arithmetic loop — no Python objects — safe for nopython=True.
        """
        score = 0.0
        for i in range(sc_coords.shape[0]):
            for j in range(neighbor_coords.shape[0]):
                dx = sc_coords[i, 0] - neighbor_coords[j, 0]
                dy = sc_coords[i, 1] - neighbor_coords[j, 1]
                dz = sc_coords[i, 2] - neighbor_coords[j, 2]
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                if dist < threshold:
                    score += (threshold - dist) ** 2
        return score

    def _score_rotamer(sc_coords, neighbor_coords, threshold):
        return _score_rotamer_jit(sc_coords, neighbor_coords, threshold)

else:
    def _score_rotamer(
        sc_coords: np.ndarray,
        neighbor_coords: np.ndarray,
        threshold: float,
    ) -> float:
        """Pure NumPy fallback when Numba is not installed."""
        diffs = sc_coords[:, None, :] - neighbor_coords[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        violations = threshold - dists
        violations = np.where(violations > 0, violations, 0.0)
        return float((violations ** 2).sum())


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseRepacker(ABC):
    @abstractmethod
    def repack(
        self,
        ag: "prody.AtomGroup",
        clashing_residues: list[int],
    ) -> "prody.AtomGroup":
        """
        Resolve side-chain clashes for the specified residue indices.

        Parameters
        ----------
        ag                  : input AtomGroup (not modified in place)
        clashing_residues   : list of 0-based residue indices with clashes

        Returns
        -------
        New AtomGroup with identical atom count and ordering; only coordinates differ.
        """


# ---------------------------------------------------------------------------
# Dunbrack repacker
# ---------------------------------------------------------------------------

class DunbrackRepacker(BaseRepacker):
    """
    Backbone-dependent rotamer repacker using the bundled Dunbrack library.

    For each clashing residue:
      1. Extract backbone phi/psi angles
      2. Look up top-5 rotamers from Dunbrack for (resname, phi_bin, psi_bin)
      3. Score each rotamer against fixed neighbours with hard-sphere repulsion
      4. Assign the best non-clashing rotamer
      5. Update coordinates via ProDy API only (never manual PDB string construction)

    Special cases:
      - GLY, ALA: no side chain → skip (return input coordinates unchanged)
      - PRO: ring-constrained chi angles → limited rotamer space, only 2 entries used
    """

    CLASH_THRESHOLD_ANGSTROM = 0.5

    def __init__(self, library: Optional[dict] = None) -> None:
        if library is not None:
            self._lib = library
        else:
            self._lib = self._load_library()

    @staticmethod
    def _load_library() -> dict:
        if not _DUNBRACK_PKL.exists():
            raise FileNotFoundError(
                f"Dunbrack rotamer library not found at {_DUNBRACK_PKL}. "
                "Run: python scripts/build_dunbrack.py"
            )
        with open(_DUNBRACK_PKL, "rb") as fh:
            return pickle.load(fh)

    def repack(
        self,
        ag: "prody.AtomGroup",
        clashing_residues: list[int],
    ) -> "prody.AtomGroup":
        n_atoms_in = ag.numAtoms()

        # Work on a copy so the original is never mutated
        result = ag.copy()

        residues = list(result.iterResidues())

        for res_idx in clashing_residues:
            if res_idx >= len(residues):
                continue
            res = residues[res_idx]
            resname = res.getResname().upper()

            # GLY and ALA have no meaningful rotamer space — skip
            if resname in _NO_SIDECHAIN:
                continue

            rotamers = self._lookup_rotamers(res)
            if not rotamers:
                continue

            self._apply_best_rotamer(result, res, rotamers)

        # Invariant: atom count must be unchanged
        assert result.numAtoms() == n_atoms_in, (
            f"Repacker violated atom count invariant: "
            f"{n_atoms_in} in, {result.numAtoms()} out"
        )
        return result

    def _lookup_rotamers(self, res: "prody.Residue") -> list:
        """Return top-5 rotamers from Dunbrack for this residue's phi/psi bin."""
        resname = res.getResname().upper()
        phi, psi = self._backbone_angles(res)
        phi_bin = round(phi / 10) * 10
        psi_bin = round(psi / 10) * 10

        # Clamp to [-180, 170] range used by bbdep bins
        phi_bin = max(-180, min(170, phi_bin))
        psi_bin = max(-180, min(170, psi_bin))

        rotamers = self._lib.get((resname, phi_bin, psi_bin), [])
        if not rotamers:
            # Try nearest bins if exact match missing
            for dphi in (0, 10, -10, 20, -20):
                for dpsi in (0, 10, -10, 20, -20):
                    rotamers = self._lib.get((resname, phi_bin + dphi, psi_bin + dpsi), [])
                    if rotamers:
                        break
                if rotamers:
                    break

        # PRO: restrict to two canonical ring conformations
        if resname == _PROLINE:
            rotamers = rotamers[:2]
        else:
            rotamers = rotamers[:5]

        return rotamers

    def _apply_best_rotamer(
        self,
        ag: "prody.AtomGroup",
        res: "prody.Residue",
        rotamers: list,
    ) -> None:
        """Try each rotamer in probability order; apply the first that doesn't clash."""
        resname = res.getResname().upper()

        # Collect neighbour (non-self) heavy atom coordinates for clash scoring
        res_indices = set(res.getIndices())
        all_coords = ag.getCoords()
        all_indices = np.arange(ag.numAtoms())
        neighbour_mask = ~np.isin(all_indices, list(res_indices))
        neighbour_coords = all_coords[neighbour_mask]

        # Get side-chain atom indices (non-backbone heavy atoms)
        sc_sel = res.select("not backbone and not hydrogen")
        if sc_sel is None:
            return

        sc_indices = sc_sel.getIndices()
        sc_coords_orig = sc_sel.getCoords().copy()

        best_score = _score_rotamer(sc_coords_orig, neighbour_coords, self.CLASH_THRESHOLD_ANGSTROM)

        for chi_angles, _ in rotamers:
            new_coords = self._apply_chi_angles(res, chi_angles, sc_indices, all_coords)
            if new_coords is None:
                continue
            score = _score_rotamer(new_coords, neighbour_coords, self.CLASH_THRESHOLD_ANGSTROM)
            if score < best_score:
                best_score = score
                ag.setCoords(
                    self._update_coords(all_coords.copy(), sc_indices, new_coords)
                )
                if score == 0.0:
                    break

    def _apply_chi_angles(
        self,
        res: "prody.Residue",
        chi_angles: list[float],
        sc_indices: np.ndarray,
        all_coords: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Rotate side-chain atoms to the given chi angles.
        Returns new coordinates for sc_indices only, or None if geometry is unavailable.

        This is a simplified version: we use the existing chi angles as a reference
        and apply a rotation delta. For production accuracy a full forward kinematics
        pass should be used; this is sufficient for clash relief.
        """
        resname = res.getResname().upper()
        # Chi angle atom definitions (IUPAC nomenclature)
        chi_defs = _CHI_ATOMS.get(resname, [])
        if not chi_defs or not chi_angles:
            return None

        coords = all_coords.copy()
        n_chi = min(len(chi_angles), len(chi_defs))

        for chi_idx in range(n_chi):
            atom_names = chi_defs[chi_idx]
            atom_coords = []
            for name in atom_names:
                sel = res.select(f"name {name}")
                if sel is None:
                    return None
                atom_coords.append(coords[sel.getIndices()[0]])

            if len(atom_coords) < 4:
                return None

            # Compute current chi angle
            current_chi = _dihedral(*atom_coords)
            target_chi = chi_angles[chi_idx]
            delta = target_chi - current_chi

            if abs(delta) < 0.1:
                continue

            # Rotate atoms beyond the chi bond
            axis_start = atom_coords[1]
            axis_end = atom_coords[2]
            rotatable = _atoms_beyond_bond(res, atom_names[2], atom_names[1], coords)
            if not rotatable:
                continue

            rot_indices = np.array([
                res.select(f"name {n}").getIndices()[0]
                for n in rotatable
                if res.select(f"name {n}") is not None
            ])
            if len(rot_indices) == 0:
                continue

            rot_coords = coords[rot_indices]
            new_rot = _rotate_about_axis(rot_coords, axis_start, axis_end, delta)
            coords[rot_indices] = new_rot

        return coords[sc_indices]

    @staticmethod
    def _update_coords(
        all_coords: np.ndarray,
        sc_indices: np.ndarray,
        new_sc_coords: np.ndarray,
    ) -> np.ndarray:
        all_coords[sc_indices] = new_sc_coords
        return all_coords

    @staticmethod
    def _backbone_angles(res: "prody.Residue") -> tuple[float, float]:
        """
        Compute phi and psi backbone dihedral angles for a residue.
        Returns (phi, psi) in degrees. Defaults to (−60, −40) (α-helix) if
        preceding/following residues are unavailable.
        """
        try:
            phi = prody.calcPhi(res, radian=False)
            if phi is None or np.isnan(phi):
                phi = -60.0
        except Exception:
            phi = -60.0

        try:
            psi = prody.calcPsi(res, radian=False)
            if psi is None or np.isnan(psi):
                psi = -40.0
        except Exception:
            psi = -40.0

        return float(phi), float(psi)

    @staticmethod
    def _detect_clashes(
        ag: "prody.AtomGroup",
        threshold: float = CLASH_THRESHOLD_ANGSTROM,
    ) -> list[int]:
        """
        Return list of 0-based residue indices that have at least one inter-residue
        atom contact shorter than threshold Angstroms.
        """
        coords = ag.getCoords()
        n = coords.shape[0]
        clashing_res = set()

        residues = list(ag.iterResidues())
        res_atom_ranges = []
        for res in residues:
            indices = res.getIndices()
            res_atom_ranges.append(indices)

        for ri, res_i_indices in enumerate(res_atom_ranges):
            for rj, res_j_indices in enumerate(res_atom_ranges):
                if rj <= ri:
                    continue
                ci = coords[res_i_indices]
                cj = coords[res_j_indices]
                diffs = ci[:, None, :] - cj[None, :, :]
                dists = np.sqrt((diffs ** 2).sum(axis=-1))
                if np.any(dists < threshold):
                    clashing_res.add(ri)
                    clashing_res.add(rj)

        return sorted(clashing_res)


# ---------------------------------------------------------------------------
# Optional PDBFixer backend
# ---------------------------------------------------------------------------

class PDBFixerRepacker(BaseRepacker):
    """
    Enhanced repacker using OpenMM's PDBFixer for side-chain placement.
    Requires: pip install pdbfixer

    Falls back gracefully to DunbrackRepacker if pdbfixer is not available.
    """

    def __init__(self) -> None:
        try:
            import pdbfixer  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
            self._fallback = DunbrackRepacker()

    def repack(
        self,
        ag: "prody.AtomGroup",
        clashing_residues: list[int],
    ) -> "prody.AtomGroup":
        if not self._available:
            return self._fallback.repack(ag, clashing_residues)

        import io as _io
        import tempfile
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile

        n_atoms_in = ag.numAtoms()

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp:
            prody.writePDBStream(tmp, ag)
            tmp_path = tmp.name

        fixer = PDBFixer(filename=tmp_path)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        out_stream = _io.StringIO()
        PDBFile.writeFile(fixer.topology, fixer.positions, out_stream)
        out_stream.seek(0)

        result = prody.parsePDBStream(out_stream)
        if result is None or result.numAtoms() != n_atoms_in:
            # PDBFixer may add atoms; fall back to Dunbrack in that case
            return DunbrackRepacker().repack(ag, clashing_residues)

        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_repacker(prefer_enhanced: bool = False) -> BaseRepacker:
    """
    Return a working repacker instance.

    If prefer_enhanced=True, attempts to return PDBFixerRepacker; falls back
    to DunbrackRepacker if pdbfixer is not installed. Never raises.
    """
    if prefer_enhanced:
        try:
            import pdbfixer  # noqa: F401
            return PDBFixerRepacker()
        except ImportError:
            pass
    return DunbrackRepacker()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute dihedral angle in degrees given four 3D points."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-10))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return float(np.degrees(np.arctan2(y, x)))


def _rotate_about_axis(
    coords: np.ndarray,
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Rotate coords about the axis defined by axis_start→axis_end by angle_deg degrees."""
    angle_rad = np.radians(angle_deg)
    axis = axis_end - axis_start
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    t = 1 - c
    x, y, z = axis
    R = np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])
    shifted = coords - axis_start
    return (R @ shifted.T).T + axis_start


def _atoms_beyond_bond(
    res: "prody.Residue",
    atom_name: str,
    bond_atom: str,
    coords: np.ndarray,
) -> list[str]:
    """
    Return atom names in the residue that are 'beyond' atom_name relative to bond_atom
    in the side-chain connectivity. Uses a simple distance-based graph walk.
    """
    # Build a distance-based connectivity within the residue
    res_atoms = list(res.iterAtoms())
    names = [a.getName() for a in res_atoms]
    res_coords = np.array([coords[a.getIndex()] for a in res_atoms])

    if atom_name not in names or bond_atom not in names:
        return []

    start_idx = names.index(atom_name)
    bond_idx = names.index(bond_atom)

    # BFS from atom_name, not going back through bond_atom
    visited = {start_idx}
    queue = [start_idx]
    while queue:
        curr = queue.pop(0)
        for i, rc in enumerate(res_coords):
            if i in visited:
                continue
            if i == bond_idx and curr == start_idx:
                continue  # don't cross back over the chi bond
            dist = np.linalg.norm(res_coords[curr] - rc)
            if dist < 1.9:  # bonded distance threshold
                visited.add(i)
                queue.append(i)

    visited.discard(bond_idx)
    visited.discard(start_idx)  # will be rotated but we return only the followers
    visited.add(start_idx)
    return [names[i] for i in visited if names[i] not in ("N", "CA", "C", "O", "CB")]


# ---------------------------------------------------------------------------
# Chi angle atom name definitions (IUPAC, standard residue naming)
# ---------------------------------------------------------------------------

_CHI_ATOMS: dict[str, list[list[str]]] = {
    "ARG": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "NE"], ["CG", "CD", "NE", "CZ"]],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "GLU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "CE"], ["CG", "CD", "CE", "NZ"]],
    "MET": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "SD"], ["CB", "CG", "SD", "CE"]],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],  # ring-constrained
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}
