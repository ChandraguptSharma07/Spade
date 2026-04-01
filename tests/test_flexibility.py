"""Tests for spade.core.flexibility."""

import numpy as np

from spade.core.flexibility import (
    TIER_DISORDERED,
    TIER_FLEXIBLE,
    TIER_MODERATE,
    TIER_RIGID,
    FlexibilityProfile,
    build_flexibility_graph,
    build_flexibility_profile,
    classify_residues,
    compute_mode_weights,
)


# ---------------------------------------------------------------------------
# classify_residues
# ---------------------------------------------------------------------------

class TestClassifyResidues:
    def test_rigid(self):
        tiers = classify_residues(np.array([95.0]))
        assert tiers[0] == TIER_RIGID

    def test_moderate(self):
        tiers = classify_residues(np.array([80.0]))
        assert tiers[0] == TIER_MODERATE

    def test_flexible(self):
        tiers = classify_residues(np.array([60.0]))
        assert tiers[0] == TIER_FLEXIBLE

    def test_disordered(self):
        tiers = classify_residues(np.array([30.0]))
        assert tiers[0] == TIER_DISORDERED

    def test_boundary_rigid(self):
        # Exactly 90 → rigid
        tiers = classify_residues(np.array([90.0]))
        assert tiers[0] == TIER_RIGID

    def test_boundary_moderate(self):
        tiers = classify_residues(np.array([70.0]))
        assert tiers[0] == TIER_MODERATE

    def test_boundary_flexible(self):
        tiers = classify_residues(np.array([50.0]))
        assert tiers[0] == TIER_FLEXIBLE

    def test_mixed_array(self):
        plddt = np.array([95.0, 80.0, 60.0, 30.0])
        tiers = classify_residues(plddt)
        assert tiers[0] == TIER_RIGID
        assert tiers[1] == TIER_MODERATE
        assert tiers[2] == TIER_FLEXIBLE
        assert tiers[3] == TIER_DISORDERED

    def test_all_residues_classified(self):
        plddt = np.random.uniform(0, 100, size=50)
        tiers = classify_residues(plddt)
        assert len(tiers) == 50
        assert all(v in (TIER_RIGID, TIER_MODERATE, TIER_FLEXIBLE, TIER_DISORDERED)
                   for v in tiers.values())


# ---------------------------------------------------------------------------
# build_flexibility_graph
# ---------------------------------------------------------------------------

def _make_coords(n_res: int, spacing: float = 4.0) -> np.ndarray:
    """Create a linear chain of Cα coordinates spaced evenly along x-axis."""
    coords = np.zeros((n_res, 3), dtype=float)
    coords[:, 0] = np.arange(n_res) * spacing
    return coords


class TestBuildFlexibilityGraph:
    def test_output_shape(self):
        n_res = 6
        pocket = np.array([2, 3])
        pae = np.ones((n_res, n_res), dtype=np.float32) * 3.0
        np.fill_diagonal(pae, 0.0)
        # Spacing 4 Å → residues 0-5, pocket at 8-12 Å; cutoff 12 Å covers 0..5
        coords = _make_coords(n_res, spacing=3.0)
        graph, _ = build_flexibility_graph(pae, pocket, coords, cutoff_angstrom=12.0)
        assert graph.shape == (len(pocket), len(pocket))

    def test_low_pae_gives_positive_weight(self):
        n_res = 4
        pocket = np.array([1, 2])
        pae = np.full((n_res, n_res), 8.0, dtype=np.float32)
        np.fill_diagonal(pae, 0.0)
        pae[1, 2] = pae[2, 1] = 2.0   # very low PAE between pocket residues
        coords = _make_coords(n_res, spacing=3.0)
        graph, _ = build_flexibility_graph(pae, pocket, coords, cutoff_angstrom=20.0, pae_flexible_threshold=5.0)
        # pocket indices 0↔1 correspond to residues 1↔2
        assert graph[0, 1] > 0
        assert graph[1, 0] > 0

    def test_high_pae_gives_zero_weight(self):
        n_res = 4
        pocket = np.array([1, 2])
        pae = np.full((n_res, n_res), 20.0, dtype=np.float32)
        np.fill_diagonal(pae, 0.0)
        coords = _make_coords(n_res, spacing=3.0)
        graph, _ = build_flexibility_graph(
            pae, pocket, coords, cutoff_angstrom=20.0, pae_flexible_threshold=5.0
        )
        assert np.all(graph == 0.0)

    # ------------------------------------------------------------------
    # MANDATORY TRAP TEST: inter-domain PAE must not leak into graph
    # ------------------------------------------------------------------
    def test_inter_domain_pae_trap(self):
        """
        Residues far from the pocket get zero weight in the graph regardless
        of their PAE value. This is the critical correctness invariant.

        Setup: 10 residues. Pocket = {4, 5}. Cutoff = 8 Å. Spacing = 5 Å.
        Residue 4 is at x=20, residue 5 at x=25.
        Residues 0-2 are at x=0,5,10 — all >8 Å from the pocket.
        We assign them very low PAE (looks flexible) to test that they
        still get zero weight in the graph.
        """
        n_res = 10
        pocket = np.array([4, 5])
        pae = np.full((n_res, n_res), 20.0, dtype=np.float32)
        np.fill_diagonal(pae, 0.0)

        # Give far-away residues (0,1,2) suspiciously low inter-domain PAE
        pae[0, 1] = pae[1, 0] = 1.0
        pae[0, 2] = pae[2, 0] = 1.0
        pae[1, 2] = pae[2, 1] = 1.0

        # Give pocket residues reasonable PAE
        pae[4, 5] = pae[5, 4] = 3.0

        coords = _make_coords(n_res, spacing=5.0)  # residues 0-3 at x=0..15, pocket at x=20,25

        graph, inter_domain_warning = build_flexibility_graph(
            pae, pocket, coords, cutoff_angstrom=8.0, pae_flexible_threshold=5.0
        )

        # Graph covers pocket residues only — shape (2, 2)
        assert graph.shape == (2, 2)

        # The pocket-residue pair (4,5) is within neighbourhood → should have weight
        assert graph[0, 1] > 0, "Pocket residue pair should have positive weight"

        # Residues 0,1,2 never appear in the graph at all (wrong shape would catch this)
        # The graph is 2×2, so out-of-pocket residues have no entries by construction.

        # The inter-domain warning should fire because far residues have low PAE
        assert inter_domain_warning is True, (
            "inter_domain_pae_warning should be True when out-of-pocket residues "
            "have suspiciously low PAE"
        )

    def test_no_warning_when_only_pocket_has_low_pae(self):
        n_res = 6
        pocket = np.array([2, 3])
        pae = np.full((n_res, n_res), 15.0, dtype=np.float32)
        np.fill_diagonal(pae, 0.0)
        # Only pocket residues have low PAE
        pae[2, 3] = pae[3, 2] = 2.0
        coords = _make_coords(n_res, spacing=3.0)
        _, inter_domain_warning = build_flexibility_graph(
            pae, pocket, coords, cutoff_angstrom=20.0, pae_flexible_threshold=5.0
        )
        # All residues are within 20 Å so there are no "out-of-pocket" residues → no warning
        assert inter_domain_warning is False


# ---------------------------------------------------------------------------
# build_flexibility_profile  (integration)
# ---------------------------------------------------------------------------

class TestBuildFlexibilityProfile:
    def _make_inputs(self, n_res=8):
        plddt = np.array([95, 92, 85, 75, 65, 55, 40, 30], dtype=float)[:n_res]
        pae = np.random.uniform(1, 10, (n_res, n_res)).astype(np.float32)
        np.fill_diagonal(pae, 0)
        pocket = np.array([3, 4, 5])
        coords = _make_coords(n_res, spacing=3.5)
        return plddt, pae, pocket, coords

    def test_returns_profile(self):
        plddt, pae, pocket, coords = self._make_inputs()
        profile = build_flexibility_profile(plddt, pae, pocket, coords)
        assert isinstance(profile, FlexibilityProfile)

    def test_disordered_residues_identified(self):
        plddt = np.array([95.0, 80.0, 60.0, 30.0, 25.0])
        pae = np.zeros((5, 5), dtype=np.float32)
        pocket = np.array([0, 1])
        coords = _make_coords(5, spacing=3.0)
        profile = build_flexibility_profile(plddt, pae, pocket, coords)
        assert 3 in profile.disordered_residues
        assert 4 in profile.disordered_residues

    def test_mode_weight_vector_shape(self):
        plddt, pae, pocket, coords = self._make_inputs()
        profile = build_flexibility_profile(plddt, pae, pocket, coords)
        assert profile.mode_weight_vector.shape == (len(plddt),)

    def test_flexible_residues_get_highest_weight(self):
        # Residue 0=rigid, 1=flexible → flexible should have higher weight
        plddt = np.array([95.0, 60.0])
        pae = np.zeros((2, 2), dtype=np.float32)
        pocket = np.array([0, 1])
        coords = _make_coords(2, spacing=3.0)
        profile = build_flexibility_profile(plddt, pae, pocket, coords)
        assert profile.mode_weight_vector[1] > profile.mode_weight_vector[0]

    def test_graph_shape_matches_pocket(self):
        plddt, pae, pocket, coords = self._make_inputs()
        profile = build_flexibility_profile(plddt, pae, pocket, coords)
        n_p = len(pocket)
        assert profile.flexibility_graph.shape == (n_p, n_p)


# ---------------------------------------------------------------------------
# compute_mode_weights
# ---------------------------------------------------------------------------

class TestComputeModeWeights:
    def _make_profile(self, n_res=10, n_pocket=3):
        plddt = np.random.uniform(50, 90, n_res)
        pae = np.random.uniform(1, 8, (n_res, n_res)).astype(np.float32)
        np.fill_diagonal(pae, 0)
        pocket = np.arange(n_pocket)
        coords = _make_coords(n_res, spacing=3.0)
        return build_flexibility_profile(plddt, pae, pocket, coords)

    def test_output_length(self):
        profile = self._make_profile()
        weights = compute_mode_weights(profile, n_modes=20)
        assert len(weights) == 20

    def test_truncates_when_vector_longer(self):
        profile = self._make_profile(n_res=30)
        weights = compute_mode_weights(profile, n_modes=5)
        assert len(weights) == 5

    def test_pads_when_vector_shorter(self):
        profile = self._make_profile(n_res=3)
        weights = compute_mode_weights(profile, n_modes=20)
        assert len(weights) == 20
        # Padded entries should be small but non-zero
        assert all(w >= 0 for w in weights)
