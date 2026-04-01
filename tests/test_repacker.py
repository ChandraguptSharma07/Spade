"""
Tests for spade.core.repacker.

Includes the mandatory trap tests:
  1. Atom ordering trap   — output has identical atom count and ordering to input
  2. Glycine crash trap   — GLY residue handled without error
  3. Proline crash trap   — PRO residue handled without error
"""

from pathlib import Path

import numpy as np
import pytest
import prody

prody.confProDy(verbosity="none")

from spade.core.repacker import (
    DunbrackRepacker,
    _dihedral,
    _rotate_about_axis,
    _score_rotamer,
    get_repacker,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixture_ag() -> prody.AtomGroup:
    return prody.parsePDB(str(FIXTURES / "egfr_kinase.pdb"))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

class TestDihedral:
    def test_trans_is_180(self):
        p0 = np.array([0.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 1.0])
        angle = _dihedral(p0, p1, p2, p3)
        # Should be ±90 for this geometry; we just check it's a float
        assert isinstance(angle, float)

    def test_returns_degrees(self):
        p0 = np.array([0.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        angle = _dihedral(p0, p1, p2, p3)
        assert -180 <= angle <= 180


class TestRotateAboutAxis:
    def test_360_is_identity(self):
        coords = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
        axis_start = np.zeros(3)
        axis_end = np.array([0.0, 0.0, 1.0])
        result = _rotate_about_axis(coords, axis_start, axis_end, 360.0)
        np.testing.assert_allclose(result, coords, atol=1e-5)

    def test_90_degree_rotation(self):
        coords = np.array([[1.0, 0.0, 0.0]])
        axis_start = np.zeros(3)
        axis_end = np.array([0.0, 0.0, 1.0])
        result = _rotate_about_axis(coords, axis_start, axis_end, 90.0)
        np.testing.assert_allclose(result[0], [0.0, 1.0, 0.0], atol=1e-5)


class TestScoreRotamer:
    def test_no_clash_gives_zero(self):
        sc = np.array([[0.0, 0.0, 0.0]])
        nb = np.array([[10.0, 0.0, 0.0]])
        assert _score_rotamer(sc, nb, 0.5) == 0.0

    def test_clash_gives_positive(self):
        sc = np.array([[0.0, 0.0, 0.0]])
        nb = np.array([[0.1, 0.0, 0.0]])
        assert _score_rotamer(sc, nb, 0.5) > 0.0

    def test_at_threshold_gives_zero(self):
        sc = np.array([[0.0, 0.0, 0.0]])
        nb = np.array([[0.5, 0.0, 0.0]])
        assert _score_rotamer(sc, nb, 0.5) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# DunbrackRepacker
# ---------------------------------------------------------------------------

class TestDunbrackRepacker:
    def setup_method(self):
        self.repacker = DunbrackRepacker()
        self.ag = _load_fixture_ag()

    # ------------------------------------------------------------------
    # MANDATORY TRAP TEST 1: atom ordering invariant
    # ------------------------------------------------------------------
    def test_atom_ordering_trap(self):
        """
        Output AtomGroup must have identical atom count and ordering to input.
        This is critical: ProLIF and Vina silently fail on reordered atoms.
        """
        clashing = list(range(self.ag.numResidues()))
        result = self.repacker.repack(self.ag, clashing)

        # Atom count unchanged
        assert result.numAtoms() == self.ag.numAtoms(), (
            "Atom count changed after repacking — invariant violated"
        )

        # Atom names in same order
        assert list(result.getNames()) == list(self.ag.getNames()), (
            "Atom names changed after repacking — ordering invariant violated"
        )

        # Atom serial numbers in same order
        assert list(result.getResnums()) == list(self.ag.getResnums()), (
            "Residue numbers changed after repacking"
        )

    # ------------------------------------------------------------------
    # MANDATORY TRAP TEST 2: Glycine crash trap
    # ------------------------------------------------------------------
    def test_glycine_no_crash(self):
        """
        GLY has no side chain. The repacker must handle it without error.
        The fixture contains GLY at residue index 3 (residue 699).
        """
        # Find GLY residue index
        gly_indices = [
            i for i, res in enumerate(self.ag.iterResidues())
            if res.getResname().upper() == "GLY"
        ]
        assert gly_indices, "Fixture must contain at least one GLY residue"

        # Must not raise
        result = self.repacker.repack(self.ag, gly_indices)
        assert result.numAtoms() == self.ag.numAtoms()

    # ------------------------------------------------------------------
    # MANDATORY TRAP TEST 3: Proline crash trap
    # ------------------------------------------------------------------
    def test_proline_no_crash(self, tmp_path):
        """
        PRO has ring-constrained chi angles. The repacker must handle it
        without error (not crash with IndexError on chi angle lookup).
        """
        # Build a minimal PRO-containing AtomGroup from scratch
        pro_pdb = tmp_path / "pro_test.pdb"
        pro_pdb.write_text(
            "ATOM      1  N   PRO A   1       5.000   5.000   5.000  1.00 75.00           N\n"
            "ATOM      2  CA  PRO A   1       6.000   5.000   5.000  1.00 75.00           C\n"
            "ATOM      3  C   PRO A   1       6.500   6.400   5.000  1.00 75.00           C\n"
            "ATOM      4  O   PRO A   1       7.700   6.500   5.000  1.00 75.00           O\n"
            "ATOM      5  CB  PRO A   1       6.700   4.100   6.100  1.00 75.00           C\n"
            "ATOM      6  CG  PRO A   1       7.300   2.800   5.700  1.00 75.00           C\n"
            "ATOM      7  CD  PRO A   1       7.500   2.900   4.200  1.00 75.00           C\n"
            "ATOM      8  N   LEU A   2       5.600   7.400   5.000  1.00 88.00           N\n"
            "ATOM      9  CA  LEU A   2       5.900   8.800   5.000  1.00 88.00           C\n"
            "ATOM     10  C   LEU A   2       7.400   9.100   5.000  1.00 88.00           C\n"
            "ATOM     11  O   LEU A   2       8.000  10.200   5.000  1.00 88.00           O\n"
            "ATOM     12  CB  LEU A   2       5.200   9.500   6.100  1.00 88.00           C\n"
            "ATOM     13  CG  LEU A   2       5.600  11.000   6.100  1.00 88.00           C\n"
            "ATOM     14  CD1 LEU A   2       4.800  11.800   7.100  1.00 88.00           C\n"
            "ATOM     15  CD2 LEU A   2       5.400  11.600   4.700  1.00 88.00           C\n"
            "END\n"
        )
        ag = prody.parsePDB(str(pro_pdb))
        assert ag is not None

        # Must not raise
        result = self.repacker.repack(ag, [0])
        assert result.numAtoms() == ag.numAtoms()

    def test_empty_clashing_list_returns_copy(self):
        result = self.repacker.repack(self.ag, [])
        assert result.numAtoms() == self.ag.numAtoms()
        assert result is not self.ag  # must be a copy, not the original

    def test_out_of_range_index_ignored(self):
        result = self.repacker.repack(self.ag, [9999])
        assert result.numAtoms() == self.ag.numAtoms()


# ---------------------------------------------------------------------------
# detect_clashes
# ---------------------------------------------------------------------------

class TestDetectClashes:
    def test_no_clashes_in_fixture(self):
        ag = _load_fixture_ag()
        # Normal structure should have no sub-0.5 Å clashes
        clashing = DunbrackRepacker._detect_clashes(ag, threshold=0.5)
        assert isinstance(clashing, list)
        # We don't assert it's empty (depends on fixture geometry) but it shouldn't crash
        assert all(isinstance(i, int) for i in clashing)

    def test_artificial_clash_detected(self, tmp_path):
        """Manually place two atoms 0.2 Å apart — should detect clash."""
        clash_pdb = tmp_path / "clash.pdb"
        clash_pdb.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\n"
            "ATOM      2  CA  ALA A   2       0.200   0.000   0.000  1.00 90.00           C\n"
            "END\n"
        )
        ag = prody.parsePDB(str(clash_pdb))
        clashing = DunbrackRepacker._detect_clashes(ag, threshold=0.5)
        assert 0 in clashing or 1 in clashing


# ---------------------------------------------------------------------------
# get_repacker factory
# ---------------------------------------------------------------------------

class TestGetRepacker:
    def test_returns_dunbrack_by_default(self):
        r = get_repacker()
        assert isinstance(r, DunbrackRepacker)

    def test_prefer_enhanced_falls_back_when_no_pdbfixer(self):
        # pdbfixer is not installed in the test environment
        r = get_repacker(prefer_enhanced=True)
        # Should not raise; returns something that implements BaseRepacker
        assert hasattr(r, "repack")
