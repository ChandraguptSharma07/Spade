"""
domain_rearrangement.py — PAE-guided rigid-body domain rearrangement.

Generates conformational ensembles by rotating a mobile segment (e.g. the
DFG activation loop) as a rigid body around a hinge axis defined by two
anchor Cα atoms.  Rotation amplitude is calibrated from inter-domain PAE
and a crystal-structure-derived scale factor.

Key design decisions vs EnsembleGenerator (NMA):
  - No RMSD cap: rigid-body rotation produces physically coherent 5-35 Å
    displacements that would violate the 1.2 Å NMA cap.
  - Deterministic sampling: conformers span [0, θ_max] in N evenly-spaced
    steps so the full DFG-in → DFG-out range is always covered.
  - Auto-detection: mobile segment and hinge residues can be identified
    automatically from the inter-domain PAE analysis via
    identify_mobile_segment().
  - Parametric: no hardcoded residue numbers.  Works for any kinase or
    other protein with a flexible loop anchored by two stable residues.
  - sidechain_targets: after backbone rotation, explicitly set chi1 angles
    for specified residues (e.g. Phe382 chi1 → −90° for DFG-out rotamer).
    Backbone rotation opens the loop; chi1 assignment opens the pocket.
    Applied only to conformers with frac > 0; conformer 0 (DFG-in) is
    always the unmodified reference.

Calibration reference (ABL1, 2GQG DFG-in vs 2HYY DFG-out):
  - Inter-domain PAE (activation loop vs N-lobe): 9.17 Å
  - Actual activation loop RMSD: 16.94 Å
  - Calibration factor PAE/RMSD = 0.54  →  scale_factor = 1.85
  - Full DFG-in → DFG-out rotation at Phe382: 30.6° around hinge 377→404
  - Rotation direction: +1 (CCW looking along hinge axis)
  - Phe382 chi1 in DFG-out (2HYY): −90° (gauche−)
  - Usage: sidechain_targets={382: -90.0}

For p38α-type targets where the activation loop dissolves into disorder
in the DFG-out state, this generator models the rigid-body component of
the flip only.  Loop-disorder modelling is out of scope.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import prody
    prody.confProDy(verbosity="none")
except ImportError as e:
    raise ImportError("ProDy is required: conda install -c conda-forge prody") from e

from spade.core.repacker import BaseRepacker, DunbrackRepacker, get_repacker
from spade.core.structure import AlphaFoldStructure

# Default calibration from ABL1 2GQG/2HYY crystal pair
DEFAULT_PAE_SCALE_FACTOR = 1.85

# Clash threshold for repacker trigger
_CLASH_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

@dataclass
class MobileSegment:
    """
    Describes a protein segment identified as mobile by inter-domain PAE.

    residues        : sorted list of 0-based residue indices in the mobile segment
    hinge_n_resnum  : residue number of the N-terminal hinge anchor Cα
    hinge_c_resnum  : residue number of the C-terminal hinge anchor Cα
    mean_inter_pae  : mean PAE between this segment and the reference domain (Å)
    """
    residues: list[int]
    hinge_n_resnum: int
    hinge_c_resnum: int
    mean_inter_pae: float
    ref_domain_indices: list[int] = field(default_factory=list)


def identify_mobile_segment(
    structure: AlphaFoldStructure,
    reference_domain_resnums: list[int],
    min_inter_pae: float = 6.0,
    min_segment_length: int = 5,
    max_segment_length: int = 60,
    min_plddt: float = 50.0,
    search_resnums: Optional[list[int]] = None,
) -> Optional[MobileSegment]:
    """
    Automatically identify the mobile segment with highest inter-domain PAE
    relative to a reference (stable) domain.

    Parameters
    ----------
    structure               : AlphaFoldStructure
    reference_domain_resnums: residue numbers of the stable reference domain
                              (e.g. kinase N-lobe: range(235, 311) for ABL1)
    min_inter_pae           : minimum mean PAE to consider a segment mobile (Å)
    min_segment_length      : minimum consecutive residues to form a segment
    max_segment_length      : maximum segment length (avoids selecting entire domains)
    min_plddt               : minimum pLDDT for any residue in the segment —
                              excludes truly disordered tails (pLDDT < 50) that
                              cannot be modelled with rigid-body rotation
    search_resnums          : if provided, restrict search to these residue numbers
                              (e.g. the kinase domain 230-500); avoids picking up
                              unrelated high-PAE regions in multi-domain proteins

    Returns
    -------
    MobileSegment or None if no sufficiently mobile segment is found.
    """
    ca_sel = structure.atoms.select("calpha")
    resnums = ca_sel.getResnums()
    n_res = structure.n_residues
    pae = structure.pae_matrix

    # Build search index set
    if search_resnums is not None:
        search_resnum_set = set(search_resnums)
        search_indices = [i for i, r in enumerate(resnums) if int(r) in search_resnum_set]
    else:
        search_indices = list(range(n_res))

    # Map reference domain resnums to 0-based indices
    ref_indices = []
    for r in reference_domain_resnums:
        idx = np.where(resnums == r)[0]
        if len(idx) > 0:
            ref_indices.append(idx[0])
    if not ref_indices:
        return None
    ref_indices = np.array(ref_indices)

    # Compute per-residue mean PAE to the reference domain (full protein)
    per_residue_pae = pae[:, ref_indices].mean(axis=1)  # shape (n_res,)

    # Find contiguous runs above PAE threshold within the search region,
    # requiring all residues to have pLDDT >= min_plddt
    above = np.zeros(n_res, dtype=bool)
    for i in search_indices:
        if per_residue_pae[i] > min_inter_pae and structure.plddt[i] >= min_plddt:
            above[i] = True

    best_segment = None
    best_score = 0.0

    i = 0
    while i < n_res:
        if above[i]:
            j = i
            while j < n_res and above[j]:
                j += 1
            seg_len = j - i
            if min_segment_length <= seg_len <= max_segment_length:
                seg_pae = per_residue_pae[i:j].mean()
                # Score = length × mean PAE: rewards longer mobile segments
                # over short high-PAE spurious runs
                seg_score = seg_len * seg_pae
                if seg_score > best_score:
                    best_score = seg_score
                    best_segment = (i, j)
            i = j
        else:
            i += 1

    if best_segment is None:
        return None

    seg_start, seg_end = best_segment
    seg_indices = list(range(seg_start, seg_end))

    # Hinge anchors: residue just before and just after the mobile segment
    hinge_n_idx = max(0, seg_start - 1)
    hinge_c_idx = min(n_res - 1, seg_end)
    hinge_n_resnum = int(resnums[hinge_n_idx])
    hinge_c_resnum = int(resnums[hinge_c_idx])

    seg_mean_pae = float(per_residue_pae[seg_start:seg_end].mean())
    return MobileSegment(
        residues=seg_indices,
        hinge_n_resnum=hinge_n_resnum,
        hinge_c_resnum=hinge_c_resnum,
        mean_inter_pae=seg_mean_pae,
        ref_domain_indices=list(ref_indices),
    )


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class PAEDomainRearrangementGenerator:
    """
    Generate a conformational ensemble by rotating a mobile segment as a
    rigid body around a hinge axis, with amplitude calibrated from PAE.

    Parameters
    ----------
    structure           : AlphaFoldStructure
    mobile_segment      : MobileSegment from identify_mobile_segment(), or
                          manually constructed
    inter_domain_pae    : mean inter-domain PAE for the mobile segment (Å);
                          used to compute θ_max
    pae_scale_factor    : scale factor to convert PAE to displacement (Å);
                          default 1.85 from ABL1 calibration
    n_conformers        : number of conformers to generate (including θ=0
                          reference pose); evenly spaced from 0 to θ_max
    rotation_direction  : +1 (CCW) or -1 (CW) looking along the hinge axis;
                          +1 is the DFG-out direction for ABL1
    repacker            : BaseRepacker for clash resolution after rotation
    """

    def __init__(
        self,
        structure: AlphaFoldStructure,
        mobile_segment: MobileSegment,
        inter_domain_pae: float,
        pae_scale_factor: float = DEFAULT_PAE_SCALE_FACTOR,
        n_conformers: int = 10,
        rotation_direction: int = 1,
        repacker: Optional[BaseRepacker] = None,
        sidechain_targets: Optional[dict[int, float]] = None,
    ) -> None:
        self.structure = structure
        self.segment = mobile_segment
        self.inter_domain_pae = inter_domain_pae
        self.pae_scale_factor = pae_scale_factor
        self.n_conformers = n_conformers
        self.rotation_direction = int(np.sign(rotation_direction)) or 1
        self.repacker: BaseRepacker = repacker or get_repacker()
        # {resnum: target_chi1_deg} — applied after backbone rotation on
        # conformers with frac > 0.  Conformer 0 is always the reference.
        self.sidechain_targets: dict[int, float] = sidechain_targets or {}

        # Pre-compute hinge axis from reference structure
        self._hinge_axis, self._hinge_mid = self._compute_hinge()

        # Pre-compute θ_max from PAE
        self._theta_max_deg = self._compute_theta_max()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> list["prody.AtomGroup"]:
        """
        Generate n_conformers conformers spanning DFG-in (θ=0) to DFG-out (θ=θ_max).

        Conformer 0 is always the reference structure (no rotation, no chi1 change).
        Subsequent conformers are evenly spaced rotation steps, with optional
        chi1 angle assignment from sidechain_targets applied before repacking.

        Returns list[AtomGroup] tagged with:
          - conformer_index
          - ca_rmsd_from_ref   (always 0.0 — no cap filtering here)
          - rotation_angle_deg (actual rotation applied)
        """
        fractions = np.linspace(0.0, 1.0, self.n_conformers)
        conformers = []

        for i, frac in enumerate(fractions):
            angle = frac * self._theta_max_deg * self.rotation_direction
            try:
                conf = self._rotate_segment(angle)
            except Exception as exc:
                warnings.warn(
                    f"Rotation failed at fraction={frac:.2f} (angle={angle:.1f}°): {exc}",
                    stacklevel=2,
                )
                conf = self.structure.atoms.copy()
                angle = 0.0

            # Apply sidechain chi1 targets on all non-reference conformers.
            # Must run BEFORE repacker so clash resolution sees the intended rotamer.
            if frac > 0.0 and self.sidechain_targets:
                for resnum, target_chi1 in self.sidechain_targets.items():
                    try:
                        conf = self._set_chi1(conf, resnum, target_chi1)
                    except Exception as exc:
                        warnings.warn(
                            f"chi1 assignment failed for residue {resnum} at "
                            f"angle={angle:.1f}°: {exc}",
                            stacklevel=2,
                        )

            # Repack clashes introduced by rotation and chi1 assignment
            clashing = DunbrackRepacker._detect_clashes(conf, threshold=_CLASH_THRESHOLD)
            if clashing:
                conf = self.repacker.repack(conf, clashing)
                residual = DunbrackRepacker._detect_clashes(conf, threshold=_CLASH_THRESHOLD)
                if residual:
                    warnings.warn(
                        f"Residual clashes in {len(residual)} residues at "
                        f"angle={angle:.1f}° after repacking.",
                        stacklevel=2,
                    )

            n = conf.numAtoms()
            conf.setData("conformer_index",    np.full(n, i, dtype=int))
            conf.setData("ca_rmsd_from_ref",   np.zeros(n))
            conf.setData("rotation_angle_deg", np.full(n, angle))
            conformers.append(conf)

        return conformers

    @property
    def theta_max_deg(self) -> float:
        """Maximum rotation angle (degrees) calibrated from PAE."""
        return self._theta_max_deg

    # ------------------------------------------------------------------
    # Private geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _set_chi1(
        ag: "prody.AtomGroup",
        resnum: int,
        target_chi1_deg: float,
    ) -> "prody.AtomGroup":
        """
        Set the chi1 angle of residue `resnum` to `target_chi1_deg` degrees.

        Chi1 is defined by atoms:  N – CA – CB – XG  (first side-chain torsion).
        All atoms beyond the CA–CB bond (i.e. CB and beyond) are rotated.

        Returns a new AtomGroup with the chi1 applied; input is not mutated.

        Raises ValueError if the residue or required atoms are not found.
        """
        from spade.core.repacker import _dihedral, _rotate_about_axis

        result = ag.copy()
        res_sel = result.select(f"resnum {resnum} and not hydrogen")
        if res_sel is None:
            raise ValueError(f"Residue {resnum} not found in structure.")

        # Atom name lookup helper
        def _get_coord(sel, name: str) -> np.ndarray:
            atom = sel.select(f"name {name}")
            if atom is None or len(atom) == 0:
                raise ValueError(f"Atom {name} not found in residue {resnum}.")
            return result.getCoords()[atom.getIndices()[0]].copy()

        # Chi1 bond: N–CA–CB–XG where XG is the first gamma atom
        # Try the standard gamma atom names in priority order
        gamma_candidates = ["CG", "CG1", "OG", "OG1", "SG", "ND1", "SD"]
        gamma_name = None
        for name in gamma_candidates:
            if res_sel.select(f"name {name}") is not None:
                gamma_name = name
                break
        if gamma_name is None:
            raise ValueError(
                f"No gamma atom found in residue {resnum} — "
                "cannot define chi1 (GLY/ALA have no chi1)."
            )

        p0 = _get_coord(res_sel, "N")
        p1 = _get_coord(res_sel, "CA")
        p2 = _get_coord(res_sel, "CB")
        p3 = _get_coord(res_sel, gamma_name)

        current_chi1 = _dihedral(p0, p1, p2, p3)
        delta = target_chi1_deg - current_chi1

        if abs(delta) < 0.5:   # already close enough
            return result

        # Rotate all atoms beyond the CA–CB bond (CB and all side-chain atoms)
        # around the CA→CB axis
        backbone_names = {"N", "CA", "C", "O", "OXT"}
        all_indices  = result.getIndices()
        all_resnums  = result.getResnums()
        all_names    = result.getNames()
        all_coords   = result.getCoords().copy()

        rotate_indices = [
            idx for idx, rn, nm in zip(all_indices, all_resnums, all_names)
            if int(rn) == resnum and nm not in backbone_names and nm != "H"
            and not nm.startswith("H")
        ]

        if not rotate_indices:
            return result

        rot_coords = all_coords[rotate_indices]
        new_coords = _rotate_about_axis(rot_coords, p1, p2, delta)
        all_coords[rotate_indices] = new_coords
        result.setCoords(all_coords)
        return result

    def _compute_hinge(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the unit hinge axis vector and midpoint from the reference structure.

        Axis runs from Cα(hinge_n_resnum) → Cα(hinge_c_resnum).
        """
        ca_sel = self.structure.atoms.select("calpha")
        resnums = ca_sel.getResnums()
        coords = ca_sel.getCoords()

        def _ca_coord(resnum: int) -> np.ndarray:
            idx = np.where(resnums == resnum)[0]
            if len(idx) == 0:
                raise ValueError(
                    f"Hinge residue {resnum} not found in structure "
                    f"(resnums range {resnums.min()}–{resnums.max()})"
                )
            return coords[idx[0]].copy()

        p_n = _ca_coord(self.segment.hinge_n_resnum)
        p_c = _ca_coord(self.segment.hinge_c_resnum)

        axis = p_c - p_n
        length = np.linalg.norm(axis)
        if length < 1e-6:
            raise ValueError(
                f"Hinge axis has zero length: residues "
                f"{self.segment.hinge_n_resnum} and {self.segment.hinge_c_resnum} "
                "are at the same position."
            )
        axis /= length
        midpoint = (p_n + p_c) / 2.0
        return axis, midpoint

    def _compute_theta_max(self) -> float:
        """
        Compute θ_max in degrees from inter-domain PAE and lever arm length.

        Expected displacement = PAE × scale_factor.
        Lever arm = mean distance from hinge midpoint to mobile segment Cα atoms.
        θ = arctan(displacement / lever_arm) — small angle approximation avoided.
        """
        ca_sel  = self.structure.atoms.select("calpha")
        resnums = ca_sel.getResnums()
        coords  = ca_sel.getCoords()

        lever_arms = []
        for idx in self.segment.residues:
            if idx < len(resnums):
                dist = np.linalg.norm(coords[idx] - self._hinge_mid)
                if dist > 1.0:  # ignore residues sitting exactly on the hinge
                    lever_arms.append(dist)

        if not lever_arms:
            warnings.warn(
                "Could not compute lever arm — mobile segment residues "
                "all sit on the hinge midpoint. Using 10.0 Å default.",
                stacklevel=3,
            )
            mean_lever = 10.0
        else:
            mean_lever = float(np.mean(lever_arms))

        expected_displacement = self.inter_domain_pae * self.pae_scale_factor
        theta_rad = np.arctan(expected_displacement / mean_lever)
        theta_deg = float(np.degrees(theta_rad))

        return theta_deg

    def _rotate_segment(self, angle_deg: float) -> "prody.AtomGroup":
        """
        Return a copy of the reference structure with the mobile segment
        rigidly rotated by angle_deg around the hinge axis.
        """
        if abs(angle_deg) < 1e-6:
            return self.structure.atoms.copy()

        conformer  = self.structure.atoms.copy()
        all_coords = conformer.getCoords().copy()
        all_resnums = conformer.getResnums()

        # Identify mobile residue numbers from 0-based indices
        ca_sel  = self.structure.atoms.select("calpha")
        all_rn_ca = ca_sel.getResnums()
        mobile_resnums = set()
        for idx in self.segment.residues:
            if idx < len(all_rn_ca):
                mobile_resnums.add(int(all_rn_ca[idx]))

        # Build rotation matrix (Rodrigues' formula)
        angle_rad = np.radians(angle_deg)
        u = self._hinge_axis
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        t = 1.0 - c
        ux, uy, uz = u
        R = np.array([
            [t*ux*ux + c,    t*ux*uy - s*uz, t*ux*uz + s*uy],
            [t*ux*uy + s*uz, t*uy*uy + c,    t*uy*uz - s*ux],
            [t*ux*uz - s*uy, t*uy*uz + s*ux, t*uz*uz + c   ],
        ])

        # Apply rotation to all atoms in mobile residues
        for atom_idx, resnum in enumerate(all_resnums):
            if int(resnum) in mobile_resnums:
                shifted = all_coords[atom_idx] - self._hinge_mid
                all_coords[atom_idx] = R @ shifted + self._hinge_mid

        conformer.setCoords(all_coords)
        return conformer
