"""
Tests for spade.core.docking.

Includes mandatory trap test:
  3. Bounding box drift trap — shifting pocket residues by 2 Å must produce
     a bounding box center that differs by approximately 2 Å, not zero.
"""

import numpy as np
import pytest
import prody

prody.confProDy(verbosity="none")

from spade.core.docking import (
    BoundingBox,
    DockingResult,
    PoseResult,
    compute_bounding_box,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_atomgroup(coords: np.ndarray, name: str = "test") -> prody.AtomGroup:
    """Build a minimal ProDy AtomGroup from coordinates (all CA, one chain)."""
    n = len(coords)
    ag = prody.AtomGroup(name)
    ag.setCoords(coords)
    ag.setNames(["CA"] * n)
    ag.setResnames(["ALA"] * n)
    ag.setResnums(list(range(1, n + 1)))
    ag.setChids(["A"] * n)
    ag.setElements(["C"] * n)
    ag.setSerials(list(range(1, n + 1)))
    return ag


# ---------------------------------------------------------------------------
# BoundingBox dataclass
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_fields_accessible(self):
        bb = BoundingBox(center=np.array([1.0, 2.0, 3.0]), size=np.array([20.0, 20.0, 20.0]))
        assert bb.center.shape == (3,)
        assert bb.size.shape == (3,)


# ---------------------------------------------------------------------------
# compute_bounding_box
# ---------------------------------------------------------------------------

class TestComputeBoundingBox:
    def test_returns_bounding_box(self):
        coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=float)
        ag = _make_simple_atomgroup(coords)
        pocket = np.array([0, 1, 2])
        bb = compute_bounding_box(ag, pocket)
        assert isinstance(bb, BoundingBox)

    def test_center_is_centroid_of_pocket_residues(self):
        coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
        ag = _make_simple_atomgroup(coords)
        pocket = np.array([0, 1, 2])
        bb = compute_bounding_box(ag, pocket)
        expected = np.array([2.0, 0.0, 0.0])
        np.testing.assert_allclose(bb.center, expected, atol=1e-4)

    def test_size_at_least_15A(self):
        """Vina requires at least 15Å box per dimension."""
        coords = np.array([[0.0, 0.0, 0.0]], dtype=float)
        ag = _make_simple_atomgroup(coords)
        bb = compute_bounding_box(ag, np.array([0]))
        assert all(bb.size >= 15.0), f"Box size {bb.size} must be >= 15 Å in all dims"

    def test_size_increases_with_spread(self):
        small_coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
        large_coords = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=float)
        ag_small = _make_simple_atomgroup(small_coords)
        ag_large = _make_simple_atomgroup(large_coords)
        pocket = np.array([0, 1])
        bb_small = compute_bounding_box(ag_small, pocket, padding=0.0)
        bb_large = compute_bounding_box(ag_large, pocket, padding=0.0)
        assert all(bb_large.size >= bb_small.size)

    def test_out_of_range_residues_skipped(self):
        """Out-of-range pocket indices should not crash — fallback to all CA."""
        coords = np.array([[1.0, 2.0, 3.0]], dtype=float)
        ag = _make_simple_atomgroup(coords)
        pocket = np.array([9999])  # way out of range
        bb = compute_bounding_box(ag, pocket)
        # Should fall back to the single residue centroid
        assert isinstance(bb, BoundingBox)

    def test_empty_pocket_falls_back(self):
        coords = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=float)
        ag = _make_simple_atomgroup(coords)
        bb = compute_bounding_box(ag, np.array([], dtype=int))
        assert isinstance(bb, BoundingBox)
        assert bb.size.min() >= 15.0

    # ------------------------------------------------------------------
    # MANDATORY TRAP TEST 3: bounding box drift
    # ------------------------------------------------------------------
    def test_bounding_box_shifts_with_pocket_coordinates(self):
        """
        Shifting pocket residue Cα coords by 2 Å must produce a bounding
        box center that is ~2 Å away. A static cached box would fail this.
        """
        shift = np.array([2.0, 0.0, 0.0])
        base_coords = np.array([
            [10.0, 10.0, 10.0],
            [14.0, 10.0, 10.0],
            [12.0, 14.0, 10.0],
        ], dtype=float)

        ag_ref = _make_simple_atomgroup(base_coords, name="ref")
        ag_shifted = _make_simple_atomgroup(base_coords + shift, name="shifted")

        pocket = np.array([0, 1, 2])
        bb_ref = compute_bounding_box(ag_ref, pocket)
        bb_shifted = compute_bounding_box(ag_shifted, pocket)

        center_diff = np.linalg.norm(bb_shifted.center - bb_ref.center)
        assert center_diff == pytest.approx(2.0, abs=1e-4), (
            f"Bounding box centers should differ by 2.0 Å after 2 Å shift, "
            f"got {center_diff:.4f} Å. Is compute_bounding_box caching the result?"
        )

    def test_two_conformers_produce_different_boxes(self):
        """
        Two conformers with different pocket positions must have different boxes.
        This guards against any form of result caching across conformers.
        """
        coords_a = np.array([[5.0, 5.0, 5.0], [9.0, 5.0, 5.0]], dtype=float)
        coords_b = np.array([[15.0, 15.0, 15.0], [19.0, 15.0, 15.0]], dtype=float)
        ag_a = _make_simple_atomgroup(coords_a, "conf_a")
        ag_b = _make_simple_atomgroup(coords_b, "conf_b")
        pocket = np.array([0, 1])
        bb_a = compute_bounding_box(ag_a, pocket)
        bb_b = compute_bounding_box(ag_b, pocket)
        assert not np.allclose(bb_a.center, bb_b.center), (
            "Different conformers must produce different bounding box centers"
        )

    def test_padding_expands_box(self):
        coords = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=float)
        ag = _make_simple_atomgroup(coords)
        pocket = np.array([0, 1])
        bb_small = compute_bounding_box(ag, pocket, padding=0.0)
        bb_large = compute_bounding_box(ag, pocket, padding=10.0)
        # With extra padding, all dimensions must be larger (or at least not smaller)
        assert all(bb_large.size >= bb_small.size)


# ---------------------------------------------------------------------------
# PoseResult and DockingResult dataclasses
# ---------------------------------------------------------------------------

class TestPoseResult:
    def test_fields(self):
        coords = np.zeros((10, 3), dtype=np.float32)
        p = PoseResult(pose_index=0, score_kcal_mol=-9.5, coordinates=coords, conformer_index=2)
        assert p.pose_index == 0
        assert p.score_kcal_mol == -9.5
        assert p.coordinates.shape == (10, 3)
        assert p.conformer_index == 2


class TestDockingResult:
    def test_fields(self):
        bb = BoundingBox(center=np.zeros(3), size=np.full(3, 20.0))
        dr = DockingResult(
            conformer_index=1,
            conformer_ca_rmsd=0.8,
            poses=[],
            bounding_box=bb,
            docking_time_seconds=3.14,
        )
        assert dr.conformer_index == 1
        assert dr.conformer_ca_rmsd == pytest.approx(0.8)
        assert dr.poses == []
        assert dr.docking_time_seconds == pytest.approx(3.14)
