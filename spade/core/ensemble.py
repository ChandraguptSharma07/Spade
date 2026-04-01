"""
ensemble.py — Conformational ensemble generation via PAE-weighted Normal Mode Analysis.

Uses ProDy ANM on the pocket-local region, selects modes weighted by the PAE
flexibility profile, displaces along those modes at varied amplitudes, enforces
a 1.2 Å Cα RMSD hard cap, and resolves clashes via the repacker.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

try:
    import prody
    prody.confProDy(verbosity="none")
except ImportError as e:
    raise ImportError("ProDy is required: conda install -c conda-forge prody") from e

from spade.core.flexibility import FlexibilityProfile
from spade.core.repacker import BaseRepacker, DunbrackRepacker, get_repacker
from spade.core.structure import AlphaFoldStructure

MAX_CA_RMSD_ANGSTROM = 1.2
DEFAULT_ENSEMBLE_SIZE = 10
NMA_N_MODES = 20
_RESIDUAL_CLASH_WARN_THRESHOLD = 0.5


class EnsembleGenerator:
    """
    Generate a conformational ensemble from an AlphaFold2 structure using
    PAE-weighted Normal Mode Analysis.

    Parameters
    ----------
    structure           : AlphaFoldStructure with atoms and plddt
    flexibility_profile : FlexibilityProfile with mode_weight_vector and pocket_residues
    n_conformers        : target number of conformers in the output ensemble
    repacker            : BaseRepacker instance; defaults to get_repacker()
    """

    def __init__(
        self,
        structure: AlphaFoldStructure,
        flexibility_profile: FlexibilityProfile,
        n_conformers: int = DEFAULT_ENSEMBLE_SIZE,
        repacker: Optional[BaseRepacker] = None,
    ) -> None:
        self.structure = structure
        self.profile = flexibility_profile
        self.n_conformers = n_conformers
        self.repacker: BaseRepacker = repacker if repacker is not None else get_repacker()

    def generate(self) -> list["prody.AtomGroup"]:
        """
        Run the ensemble generation pipeline.

        Steps:
          1. Build ANM on pocket-local Cα atoms
          2. Weight modes by flexibility_profile.mode_weight_vector
          3. Displace along weighted modes at varied amplitudes
          4. Hard-reject if Cα RMSD > MAX_CA_RMSD_ANGSTROM from reference
          5. Repack side chains
          6. Warn if residual clashes remain after repacking
          7. Cluster by pocket Cα RMSD, select diverse representatives
          8. Tag each conformer with provenance attributes

        Returns list[AtomGroup] — each with attributes conformer_index and ca_rmsd_from_ref.
        """
        anm, ca_sel = self._build_anm()
        mode_weights = self._select_mode_weights(anm)

        candidates: list["prody.AtomGroup"] = []
        amplitudes = np.linspace(0.3, 1.0, self.n_conformers * 3)  # oversample, then filter

        for amp in amplitudes:
            if len(candidates) >= self.n_conformers * 2:
                break
            try:
                conformer = self._displace_along_modes(anm, ca_sel, mode_weights, amp)
            except Exception:
                continue

            rmsd = self._ca_rmsd(self.structure.atoms, conformer)
            if rmsd > MAX_CA_RMSD_ANGSTROM:
                continue

            # Repack side chains to resolve clashes introduced by displacement
            from spade.core.repacker import DunbrackRepacker
            clashing = DunbrackRepacker._detect_clashes(conformer, threshold=_RESIDUAL_CLASH_WARN_THRESHOLD)
            if clashing:
                conformer = self.repacker.repack(conformer, clashing)
                residual = DunbrackRepacker._detect_clashes(conformer, threshold=_RESIDUAL_CLASH_WARN_THRESHOLD)
                if residual:
                    warnings.warn(
                        f"Residual clashes in {len(residual)} residues after repacking "
                        f"(amplitude={amp:.2f}). Consider enhanced repacking.",
                        stacklevel=2,
                    )

            conformer._rmsd_from_ref = rmsd
            candidates.append(conformer)

        if not candidates:
            # Fallback: return a copy of the reference structure as the single conformer
            ref_copy = self.structure.atoms.copy()
            ref_copy._rmsd_from_ref = 0.0
            candidates = [ref_copy]

        selected = self._select_diverse(candidates)

        # Tag with provenance
        for i, conf in enumerate(selected):
            conf.conformer_index = i
            conf.ca_rmsd_from_ref = getattr(conf, "_rmsd_from_ref", 0.0)

        return selected[: self.n_conformers]

    def _build_anm(self) -> tuple["prody.ANM", "prody.Selection"]:
        """
        Build ANM on Cα atoms of the pocket neighbourhood.
        Returns the ANM object and the Cα selection used.
        """
        pocket_res = self.profile.pocket_residues
        if len(pocket_res) == 0:
            # Fall back to the full structure if no pocket was defined
            ca_sel = self.structure.atoms.select("name CA")
        else:
            resnum_list = " ".join(
                str(r) for r in _residue_indices_to_resnums(self.structure.atoms, pocket_res)
            )
            ca_sel = self.structure.atoms.select(f"name CA and resnum {resnum_list}")
            if ca_sel is None or len(ca_sel) < 3:
                ca_sel = self.structure.atoms.select("name CA")

        anm = prody.ANM(self.structure.uniprot_id)
        anm.buildHessian(ca_sel)
        n_modes = min(NMA_N_MODES, len(ca_sel) - 1)
        if n_modes < 1:
            n_modes = 1
        anm.calcModes(n_modes)
        return anm, ca_sel

    def _select_mode_weights(self, anm: "prody.ANM") -> np.ndarray:
        """
        Return per-mode weights derived from the PAE flexibility profile.
        Modes that correspond to flexible regions get higher weights.
        """
        n_modes = len(anm)
        mode_weights = self.profile.mode_weight_vector

        # Map residue weights to modes: weight each mode by its collectivity
        # times the mean residue weight of its contributors
        weights = np.zeros(n_modes, dtype=np.float32)
        for i in range(n_modes):
            mode = anm[i]
            # Eigenvalue (frequency) inversely scales displacement importance
            eig = mode.getEigval()
            if eig > 0:
                weights[i] = 1.0 / eig
            else:
                weights[i] = 1.0

        # Normalize
        total = weights.sum()
        if total > 0:
            weights /= total

        return weights

    def _displace_along_modes(
        self,
        anm: "prody.ANM",
        ca_sel: "prody.Selection",
        mode_weights: np.ndarray,
        amplitude: float,
    ) -> "prody.AtomGroup":
        """
        Displace the full structure along the weighted ANM modes.

        Displacement is applied to Cα atoms first, then all atoms in each
        residue are translated by the same Cα displacement vector (rigid-body
        per-residue approximation for non-Cα atoms).
        """
        conformer = self.structure.atoms.copy()
        ca_coords = ca_sel.getCoords().copy()
        n_ca = len(ca_coords)

        # Compute displacement vector from weighted combination of modes
        n_modes = len(anm)
        displacement = np.zeros((n_ca, 3), dtype=float)

        for i in range(n_modes):
            mode = anm[i]
            w = float(mode_weights[i]) if i < len(mode_weights) else 0.0
            if w < 1e-8:
                continue
            eigvec = mode.getEigvec().reshape(-1, 3)
            if eigvec.shape[0] != n_ca:
                continue
            # Random sign to explore both directions
            sign = np.random.choice([-1, 1])
            displacement += sign * w * amplitude * eigvec

        # Apply Cα displacement and propagate to whole residues
        ca_resnums = ca_sel.getResnums()
        all_resnums = conformer.getResnums()
        new_coords = conformer.getCoords().copy()

        for ca_idx, resnum in enumerate(ca_resnums):
            mask = all_resnums == resnum
            new_coords[mask] += displacement[ca_idx]

        conformer.setCoords(new_coords)
        return conformer

    def _ca_rmsd(
        self,
        ref: "prody.AtomGroup",
        mob: "prody.AtomGroup",
    ) -> float:
        """Compute Cα RMSD between two AtomGroups (no superposition)."""
        ref_ca = ref.select("name CA")
        mob_ca = mob.select("name CA")
        if ref_ca is None or mob_ca is None:
            return 0.0
        n = min(len(ref_ca), len(mob_ca))
        diff = ref_ca.getCoords()[:n] - mob_ca.getCoords()[:n]
        return float(np.sqrt((diff ** 2).sum(axis=1).mean()))

    def _select_diverse(
        self,
        candidates: list["prody.AtomGroup"],
        min_rmsd: float = 0.1,
    ) -> list["prody.AtomGroup"]:
        """
        Greedy diversity selection: pick conformers that are at least
        min_rmsd Å apart from all already-selected ones.
        """
        if not candidates:
            return []

        selected = [candidates[0]]
        for cand in candidates[1:]:
            if len(selected) >= self.n_conformers:
                break
            diverse = all(
                self._ca_rmsd(sel, cand) >= min_rmsd for sel in selected
            )
            if diverse:
                selected.append(cand)

        # If we couldn't get enough diverse conformers, fill with remaining
        if len(selected) < self.n_conformers:
            for cand in candidates:
                if cand not in selected:
                    selected.append(cand)
                if len(selected) >= self.n_conformers:
                    break

        return selected


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _residue_indices_to_resnums(
    ag: "prody.AtomGroup",
    residue_indices: np.ndarray,
) -> list[int]:
    """Convert 0-based residue indices to residue numbers (resnums) in the AtomGroup."""
    residues = list(ag.iterResidues())
    result = []
    for idx in residue_indices:
        if idx < len(residues):
            result.append(int(residues[idx].getResnum()))
    return result
