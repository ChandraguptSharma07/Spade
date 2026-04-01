"""
flexibility.py — pLDDT residue classification and PAE-weighted flexibility graph.

The flexibility graph is strictly pocket-local: only residue pairs where BOTH
residues lie within cutoff_angstrom of any pocket residue contribute to the graph.
Inter-domain PAE is excluded regardless of its value — it reflects evolutionary
uncertainty, not biological breathing motions near the binding site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# pLDDT tier thresholds (inclusive lower, exclusive upper except top)
# ---------------------------------------------------------------------------

PLDDT_RIGID = (90.0, 100.0)        # Backbone and side chains reliable
PLDDT_MODERATE = (70.0, 90.0)      # Side-chain flexibility only
PLDDT_FLEXIBLE = (50.0, 70.0)      # Backbone perturbation allowed in NMA
PLDDT_DISORDERED = (0.0, 50.0)     # Flag; excluded from NMA perturbation

TIER_RIGID = "rigid"
TIER_MODERATE = "moderate"
TIER_FLEXIBLE = "flexible"
TIER_DISORDERED = "disordered"

# PAE value above which a residue pair outside the pocket zone triggers a warning
_INTER_DOMAIN_HIGH_PAE_THRESHOLD = 10.0


@dataclass
class FlexibilityProfile:
    """
    Holds the PAE-derived flexibility characterisation for a structure.

    flexibility_graph    — (N_pocket, N_pocket) PAE-weighted adjacency matrix.
                           Rows/cols index into pocket_residues, not the full structure.
                           Pairs outside the 12 Å pocket neighbourhood are zeroed.
    mode_weight_vector   — (N_res,) weights used to scale ANM mode contributions.
                           Indexed over ALL residues in the structure.
    inter_domain_pae_warning — True if any residue pair outside the pocket zone has
                           PAE < pae_flexible_threshold (could mistakenly look flexible).
    """

    residue_tiers: dict[int, str]
    flexibility_graph: np.ndarray
    pocket_residues: np.ndarray
    disordered_residues: np.ndarray
    mode_weight_vector: np.ndarray
    inter_domain_pae_warning: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_residues(plddt: np.ndarray) -> dict[int, str]:
    """
    Assign each residue to a pLDDT tier.

    Parameters
    ----------
    plddt : (N_res,) float array of per-residue pLDDT scores

    Returns
    -------
    dict mapping residue index (0-based) to tier label string
    """
    tiers: dict[int, str] = {}
    for i, score in enumerate(plddt):
        if score >= PLDDT_RIGID[0]:
            tiers[i] = TIER_RIGID
        elif score >= PLDDT_MODERATE[0]:
            tiers[i] = TIER_MODERATE
        elif score >= PLDDT_FLEXIBLE[0]:
            tiers[i] = TIER_FLEXIBLE
        else:
            tiers[i] = TIER_DISORDERED
    return tiers


def build_flexibility_graph(
    pae_matrix: np.ndarray,
    pocket_residues: np.ndarray,
    ca_coords: np.ndarray,
    cutoff_angstrom: float = 12.0,
    pae_flexible_threshold: float = 5.0,
) -> tuple[np.ndarray, bool]:
    """
    Build a PAE-weighted adjacency matrix restricted to the pocket neighbourhood.

    The graph covers only the N_pocket residues in pocket_residues.
    Any residue pair where EITHER residue is >cutoff_angstrom from ALL pocket
    residues receives a weight of zero, regardless of its PAE value.

    Parameters
    ----------
    pae_matrix         : (N_res, N_res) full PAE matrix
    pocket_residues    : 1D int array of 0-based residue indices defining the pocket
    ca_coords          : (N_res, 3) Cα coordinates for all residues
    cutoff_angstrom    : distance threshold for pocket neighbourhood
    pae_flexible_threshold : PAE values below this are considered flexible

    Returns
    -------
    graph : (N_pocket, N_pocket) adjacency matrix; entry = (threshold - PAE) clipped to [0, ∞)
    inter_domain_warning : bool — True if any out-of-pocket pair has PAE below threshold
    """
    n_res = pae_matrix.shape[0]
    pocket_residues = np.asarray(pocket_residues, dtype=int)

    # Determine which residues are in the pocket neighbourhood (within cutoff of any pocket res)
    pocket_ca = ca_coords[pocket_residues]  # (N_pocket, 3)
    in_neighbourhood = np.zeros(n_res, dtype=bool)

    for res_idx in range(n_res):
        diffs = pocket_ca - ca_coords[res_idx]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        if np.any(dists <= cutoff_angstrom):
            in_neighbourhood[res_idx] = True

    # Check for inter-domain warning: any out-of-pocket pair with low PAE
    out_of_pocket = np.where(~in_neighbourhood)[0]
    inter_domain_warning = False
    if len(out_of_pocket) > 0:
        oop_pae = pae_matrix[np.ix_(out_of_pocket, out_of_pocket)]
        # Exclude diagonal (self-PAE ~ 0 always)
        np.fill_diagonal(oop_pae, np.inf)
        if np.any(oop_pae < pae_flexible_threshold):
            inter_domain_warning = True

    # Build pocket-local graph: N_pocket × N_pocket
    # IMPORTANT: only residues confirmed to be in the pocket neighbourhood
    # contribute. Any residue outside the 12 Å zone gets zero weight in the
    # adjacency matrix regardless of how low its PAE value is.
    n_pocket = len(pocket_residues)
    graph = np.zeros((n_pocket, n_pocket), dtype=np.float32)

    for pi, i in enumerate(pocket_residues):
        if not in_neighbourhood[i]:
            continue
        for pj, j in enumerate(pocket_residues):
            if pi == pj:
                continue  # skip self — diagonal stays zero
            if not in_neighbourhood[j]:
                continue
            pae_val = pae_matrix[i, j]
            # Weight = flexibility signal: low PAE → high weight
            weight = max(0.0, pae_flexible_threshold - pae_val)
            graph[pi, pj] = weight

    return graph, inter_domain_warning


def compute_mode_weights(
    flexibility_profile: FlexibilityProfile,
    n_modes: int,
) -> np.ndarray:
    """
    Compute a per-mode weight vector for ANM mode selection.

    Modes are weighted by the fraction of flexible/moderate residues they
    would displace. For now we use the mode_weight_vector (per-residue)
    and return it padded/truncated to n_modes.

    Returns (n_modes,) float array — higher = more weight on that mode.
    """
    w = flexibility_profile.mode_weight_vector
    if len(w) >= n_modes:
        return w[:n_modes].astype(np.float32)
    # Pad with small baseline weight
    padded = np.full(n_modes, 0.01, dtype=np.float32)
    padded[: len(w)] = w
    return padded


def build_flexibility_profile(
    plddt: np.ndarray,
    pae_matrix: np.ndarray,
    pocket_residues: np.ndarray,
    ca_coords: np.ndarray,
    cutoff_angstrom: float = 12.0,
    pae_flexible_threshold: float = 5.0,
) -> FlexibilityProfile:
    """
    Convenience function: classify residues and build the full FlexibilityProfile.

    Parameters
    ----------
    plddt           : (N_res,) per-residue pLDDT scores
    pae_matrix      : (N_res, N_res) PAE matrix
    pocket_residues : 1D int array of pocket residue indices
    ca_coords       : (N_res, 3) Cα coordinates
    """
    tiers = classify_residues(plddt)
    disordered = np.array(
        [i for i, t in tiers.items() if t == TIER_DISORDERED], dtype=int
    )

    graph, inter_domain_warning = build_flexibility_graph(
        pae_matrix,
        pocket_residues,
        ca_coords,
        cutoff_angstrom=cutoff_angstrom,
        pae_flexible_threshold=pae_flexible_threshold,
    )

    # Per-residue weight: 1.0 for flexible, 0.5 for moderate, 0.0 for rigid/disordered
    _tier_weights = {
        TIER_FLEXIBLE: 1.0,
        TIER_MODERATE: 0.5,
        TIER_RIGID: 0.0,
        TIER_DISORDERED: 0.0,
    }
    mode_weight_vector = np.array(
        [_tier_weights[tiers[i]] for i in range(len(plddt))], dtype=np.float32
    )

    return FlexibilityProfile(
        residue_tiers=tiers,
        flexibility_graph=graph,
        pocket_residues=np.asarray(pocket_residues, dtype=int),
        disordered_residues=disordered,
        mode_weight_vector=mode_weight_vector,
        inter_domain_pae_warning=inter_domain_warning,
    )
