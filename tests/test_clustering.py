"""
Tests for spade.core.clustering.

Includes mandatory trap test:
  5. PLIF-not-RMSD trap — cluster_poses must not call any RMSD function.
     Coordinate frames differ between conformers; RMSD clustering is wrong.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spade.core.clustering import (
    ConsensusResult,
    PoseCluster,
    _classify_site_confidence,
    _coordinate_pseudofp,
    _dbscan_cluster,
    _tanimoto,
    _tanimoto_distance_matrix,
    cluster_poses,
)
from spade.core.docking import BoundingBox, DockingResult, PoseResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pose(score: float, conf_idx: int, pose_idx: int = 0, n_atoms: int = 10) -> PoseResult:
    coords = np.random.RandomState(pose_idx).randn(n_atoms, 3).astype(np.float32) * 2.0
    return PoseResult(
        pose_index=pose_idx,
        score_kcal_mol=score,
        coordinates=coords,
        conformer_index=conf_idx,
    )


def _make_dr(conf_idx: int, poses: list[PoseResult]) -> DockingResult:
    bb = BoundingBox(center=np.zeros(3), size=np.full(3, 20.0))
    return DockingResult(
        conformer_index=conf_idx,
        conformer_ca_rmsd=0.5,
        poses=poses,
        bounding_box=bb,
        docking_time_seconds=1.0,
    )


# ---------------------------------------------------------------------------
# _tanimoto
# ---------------------------------------------------------------------------

class TestTanimoto:
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 1.0, 1.0])
        assert _tanimoto(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert _tanimoto(a, b) == pytest.approx(0.0)

    def test_zero_vectors(self):
        a = np.zeros(4)
        b = np.zeros(4)
        # Both zero — by convention, identical → 1.0
        assert _tanimoto(a, b) == pytest.approx(1.0)

    def test_partial_overlap(self):
        a = np.array([1.0, 1.0, 0.0])
        b = np.array([1.0, 0.0, 1.0])
        t = _tanimoto(a, b)
        assert 0.0 < t < 1.0


# ---------------------------------------------------------------------------
# _tanimoto_distance_matrix
# ---------------------------------------------------------------------------

class TestTanimotoDistanceMatrix:
    def test_diagonal_is_zero(self):
        fps = np.eye(4, dtype=np.float32)
        dm = _tanimoto_distance_matrix(fps)
        np.testing.assert_allclose(np.diag(dm), 0.0, atol=1e-6)

    def test_symmetric(self):
        fps = np.random.RandomState(0).rand(5, 8).astype(np.float32)
        dm = _tanimoto_distance_matrix(fps)
        np.testing.assert_allclose(dm, dm.T, atol=1e-6)

    def test_values_in_range(self):
        fps = np.random.RandomState(1).rand(4, 8).astype(np.float32)
        dm = _tanimoto_distance_matrix(fps)
        assert dm.min() >= -1e-6
        assert dm.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# _dbscan_cluster
# ---------------------------------------------------------------------------

class TestDbscanCluster:
    def test_single_pose_gets_label_zero(self):
        fp = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        labels = _dbscan_cluster(fp, eps=0.3)
        assert len(labels) == 1
        assert labels[0] == 0

    def test_identical_poses_cluster_together(self):
        fp = np.tile(np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32), (3, 1))
        labels = _dbscan_cluster(fp, eps=0.3)
        assert len(set(labels)) == 1, "Identical fingerprints should be in one cluster"

    def test_dissimilar_poses_in_separate_clusters(self):
        fps = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        labels = _dbscan_cluster(fps, eps=0.01)  # very tight threshold
        assert len(set(labels)) == 4, "Orthogonal fingerprints should each be in separate clusters"

    def test_empty_array(self):
        labels = _dbscan_cluster(np.zeros((0, 4), dtype=np.float32), eps=0.3)
        assert len(labels) == 0


# ---------------------------------------------------------------------------
# _coordinate_pseudofp
# ---------------------------------------------------------------------------

class TestCoordinatePseudofp:
    def test_output_length(self):
        poses = [_make_pose(-8.0, conf_idx=0, n_atoms=12)]
        fps = _coordinate_pseudofp(poses)
        assert len(fps) == 1
        assert fps[0].shape == (64,)

    def test_normalised(self):
        poses = [_make_pose(-8.0, conf_idx=0, n_atoms=12)]
        fps = _coordinate_pseudofp(poses)
        np.testing.assert_allclose(fps[0].sum(), 1.0, atol=1e-5)

    def test_empty_coords(self):
        p = PoseResult(pose_index=0, score_kcal_mol=-5.0,
                       coordinates=np.zeros((0, 3), dtype=np.float32), conformer_index=0)
        fps = _coordinate_pseudofp([p])
        assert fps[0].shape == (64,)


# ---------------------------------------------------------------------------
# _classify_site_confidence
# ---------------------------------------------------------------------------

class TestClassifySiteConfidence:
    def test_high_confidence(self):
        assert _classify_site_confidence(0.5) == "high"

    def test_medium_confidence(self):
        assert _classify_site_confidence(1.5) == "medium"

    def test_low_confidence(self):
        assert _classify_site_confidence(3.0) == "low"

    def test_boundary_values(self):
        assert _classify_site_confidence(0.99) == "high"
        assert _classify_site_confidence(1.0) == "medium"
        assert _classify_site_confidence(2.49) == "medium"
        assert _classify_site_confidence(2.5) == "low"


# ---------------------------------------------------------------------------
# cluster_poses (integration)
# ---------------------------------------------------------------------------

class TestClusterPoses:
    def _make_simple_input(self, n_conformers=3, n_poses_per=3):
        drs = []
        for ci in range(n_conformers):
            poses = [_make_pose(-8.0 - i * 0.5, ci, i) for i in range(n_poses_per)]
            drs.append(_make_dr(ci, poses))
        return drs

    def test_returns_consensus_result(self):
        drs = self._make_simple_input()
        result = cluster_poses(drs, conformers=[], ligand_mol=None)
        assert isinstance(result, ConsensusResult)

    def test_n_total_poses_correct(self):
        drs = self._make_simple_input(n_conformers=2, n_poses_per=3)
        result = cluster_poses(drs, conformers=[], ligand_mol=None)
        assert result.n_total_poses == 6

    def test_top_cluster_is_pose_cluster(self):
        drs = self._make_simple_input()
        result = cluster_poses(drs, conformers=[], ligand_mol=None)
        assert isinstance(result.top_cluster, PoseCluster)

    def test_all_clusters_non_empty(self):
        drs = self._make_simple_input()
        result = cluster_poses(drs, conformers=[], ligand_mol=None)
        assert result.n_clusters > 0
        assert len(result.all_clusters) == result.n_clusters

    def test_pose_confidence_in_range(self):
        drs = self._make_simple_input()
        result = cluster_poses(drs, conformers=[], ligand_mol=None)
        assert 0.0 <= result.pose_confidence <= 1.0

    def test_site_confidence_valid_string(self):
        drs = self._make_simple_input()
        result = cluster_poses(drs, conformers=[], ligand_mol=None)
        assert result.site_confidence in ("high", "medium", "low")

    def test_empty_docking_results(self):
        result = cluster_poses([], conformers=[], ligand_mol=None)
        assert isinstance(result, ConsensusResult)

    def test_identical_poses_in_same_cluster(self):
        """Two identical poses (same fp) should end up in one cluster."""
        # Give both poses the exact same coordinates
        coords = np.ones((10, 3), dtype=np.float32)
        p1 = PoseResult(pose_index=0, score_kcal_mol=-9.0, coordinates=coords, conformer_index=0)
        p2 = PoseResult(pose_index=0, score_kcal_mol=-8.5, coordinates=coords, conformer_index=1)
        bb = BoundingBox(center=np.zeros(3), size=np.full(3, 20.0))
        dr1 = DockingResult(0, 0.3, [p1], bb, 1.0)
        dr2 = DockingResult(1, 0.6, [p2], bb, 1.0)
        result = cluster_poses([dr1, dr2], conformers=[], ligand_mol=None)
        assert result.n_clusters == 1, (
            "Poses with identical coordinates must cluster together"
        )

    def test_top_cluster_has_best_consensus_score(self):
        drs = self._make_simple_input(n_conformers=3, n_poses_per=2)
        result = cluster_poses(drs, conformers=[], ligand_mol=None)
        scores = [c.consensus_score for c in result.all_clusters]
        assert result.top_cluster.consensus_score == min(scores)

    # ------------------------------------------------------------------
    # MANDATORY TRAP TEST 5: PLIF not RMSD
    # ------------------------------------------------------------------
    def test_cluster_poses_does_not_call_rmsd(self):
        """
        cluster_poses must NOT call any RMSD function.
        RMSD clustering across conformers with different coordinate frames
        is mathematically incoherent. All clustering must be PLIF-based.

        We verify this by:
          1. Confirming _tanimoto (Tanimoto similarity = PLIF path) IS called.
          2. Confirming prody.calcRMSD is NOT called during clustering.
        """
        drs = self._make_simple_input(n_conformers=2, n_poses_per=2)

        with patch("spade.core.clustering._tanimoto", wraps=_tanimoto) as mock_t, \
             patch("spade.core.clustering.prody", create=True) as mock_prody:
            # prody.calcRMSD must not be invoked
            mock_prody.calcRMSD = MagicMock(side_effect=AssertionError(
                "prody.calcRMSD must not be called inside cluster_poses"
            ))
            cluster_poses(drs, conformers=[], ligand_mol=None)
            # _tanimoto must have been called — proves PLIF similarity path is active
            assert mock_t.called, "_tanimoto (PLIF-based similarity) must be called during clustering"

    def test_fraction_ensemble_uses_conformer_count(self):
        """fraction_ensemble should reflect distinct conformers in each cluster."""
        # 3 conformers, 1 pose each — all different coords → likely 3 clusters
        # But we want to verify the fraction is between 0 and 1
        drs = self._make_simple_input(n_conformers=3, n_poses_per=1)
        result = cluster_poses(drs, conformers=[], ligand_mol=None)
        for cluster in result.all_clusters:
            assert 0.0 <= cluster.fraction_ensemble <= 1.0

    def test_representative_is_best_scoring_member(self):
        """The representative pose of each cluster must be the best-scoring member."""
        drs = self._make_simple_input(n_conformers=2, n_poses_per=4)
        result = cluster_poses(drs, conformers=[], ligand_mol=None)
        for cluster in result.all_clusters:
            member_scores = [p.score_kcal_mol for p in cluster.member_poses]
            rep_score = cluster.representative_pose.score_kcal_mol
            assert rep_score == min(member_scores), (
                f"Representative should have best score; got {rep_score} vs min {min(member_scores)}"
            )
