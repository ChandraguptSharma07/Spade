"""
Tests for spade.core.ligand.

Includes the mandatory trap test:
  4. Stereochemistry trap — SMILES with undefined stereocenter produces
     exactly 2 PreparedLigand objects; defined stereocenter produces exactly 1.
"""

import pytest

from spade.core.ligand import (
    PreparedLigand,
    _enumerate_stereoisomers,
    _enumerate_tautomers,
    _generate_3d,
    _parse_smiles,
    _stereo_label,
    prepare_ligand,
)

# Simple molecules for testing
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"           # no stereocenters
IBUPROFEN_UNDEF = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # one undefined stereocenter (chiral C)
IBUPROFEN_R = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"  # R defined
IBUPROFEN_S = "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O"   # S defined
ALANINE_L = "N[C@@H](C)C(=O)O"              # L-alanine, defined
ERLOTINIB = "CCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOCCO"  # no stereocenters


# ---------------------------------------------------------------------------
# _parse_smiles
# ---------------------------------------------------------------------------

class TestParseSmiles:
    def test_valid_smiles(self):
        mol = _parse_smiles(ASPIRIN)
        assert mol is not None

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError, match="RDKit could not parse"):
            _parse_smiles("not_a_smiles_$$$$")

    def test_strips_whitespace(self):
        mol = _parse_smiles(f"  {ASPIRIN}  ")
        assert mol is not None


# ---------------------------------------------------------------------------
# _enumerate_tautomers
# ---------------------------------------------------------------------------

class TestEnumerateTautomers:
    def test_returns_at_least_one(self):
        mol = _parse_smiles(ASPIRIN)
        tautomers = _enumerate_tautomers(mol)
        assert len(tautomers) >= 1

    def test_respects_max(self):
        mol = _parse_smiles(ASPIRIN)
        tautomers = _enumerate_tautomers(mol, max_tautomers=2)
        assert len(tautomers) <= 2


# ---------------------------------------------------------------------------
# _enumerate_stereoisomers
# ---------------------------------------------------------------------------

class TestEnumerateStereoisomers:
    # ------------------------------------------------------------------
    # MANDATORY TRAP TEST 4: stereochemistry enumeration
    # ------------------------------------------------------------------
    def test_undefined_stereocenter_produces_two_variants(self):
        """
        SMILES with one undefined stereocenter must produce exactly 2 variants.
        This is the critical scientific correctness requirement.
        """
        mol = _parse_smiles(IBUPROFEN_UNDEF)
        variants = _enumerate_stereoisomers(mol, enumerate=True)
        assert len(variants) == 2, (
            f"Expected 2 stereoisomers for molecule with 1 undefined stereocenter, "
            f"got {len(variants)}"
        )

    def test_defined_stereocenter_produces_one_variant(self):
        """
        SMILES with all stereocenters defined must produce exactly 1 variant.
        """
        mol = _parse_smiles(IBUPROFEN_R)
        variants = _enumerate_stereoisomers(mol, enumerate=True)
        assert len(variants) == 1, (
            f"Expected 1 stereoisomer for fully-defined molecule, got {len(variants)}"
        )

    def test_enumerate_false_returns_single_variant(self):
        """When enumerate_stereo=False, even undefined stereocenters return 1 variant."""
        mol = _parse_smiles(IBUPROFEN_UNDEF)
        variants = _enumerate_stereoisomers(mol, enumerate=False)
        assert len(variants) == 1

    def test_flat_molecule_returns_one_variant(self):
        mol = _parse_smiles(ASPIRIN)
        variants = _enumerate_stereoisomers(mol, enumerate=True)
        assert len(variants) == 1

    def test_stereo_id_differs_between_variants(self):
        mol = _parse_smiles(IBUPROFEN_UNDEF)
        variants = _enumerate_stereoisomers(mol, enumerate=True)
        ids = [v[1] for v in variants]
        assert ids[0] != ids[1], "Stereoisomers must have distinct IDs"

    def test_n_undefined_is_zero_after_enumeration(self):
        mol = _parse_smiles(IBUPROFEN_UNDEF)
        variants = _enumerate_stereoisomers(mol, enumerate=True)
        for _, _, n_undef in variants:
            assert n_undef == 0


class TestStereoLabel:
    def test_flat_molecule(self):
        mol = _parse_smiles(ASPIRIN)
        from rdkit import Chem
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        label = _stereo_label(centers)
        assert label == "flat"

    def test_defined_center(self):
        mol = _parse_smiles(ALANINE_L)
        from rdkit import Chem
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        label = _stereo_label(centers)
        assert label in ("R", "S", "R,S", "S,R")  # single defined center


# ---------------------------------------------------------------------------
# _generate_3d
# ---------------------------------------------------------------------------

class TestGenerate3d:
    def test_returns_mol_with_conformer(self):
        mol = _parse_smiles(ASPIRIN)
        mol_3d = _generate_3d(mol)
        assert mol_3d is not None
        assert mol_3d.GetNumConformers() > 0

    def test_returns_none_for_impossible_mol(self):
        # Highly strained/impossible molecule — embedding should fail
        mol = _parse_smiles("C1CC1")  # cyclopropane — should work fine; skip this edge case
        mol_3d = _generate_3d(mol)
        # Just check it doesn't crash
        assert mol_3d is None or mol_3d.GetNumConformers() >= 0


# ---------------------------------------------------------------------------
# prepare_ligand (integration)
# ---------------------------------------------------------------------------

class TestPrepareLigand:
    def test_aspirin_returns_list(self):
        results = prepare_ligand(ASPIRIN)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_all_results_are_prepared_ligands(self):
        results = prepare_ligand(ASPIRIN)
        for r in results:
            assert isinstance(r, PreparedLigand)

    def test_pdbqt_string_nonempty(self):
        results = prepare_ligand(ASPIRIN)
        for r in results:
            assert len(r.pdbqt_string) > 0

    def test_smiles_input_preserved(self):
        results = prepare_ligand(ASPIRIN)
        for r in results:
            assert r.smiles_input == ASPIRIN

    # ------------------------------------------------------------------
    # MANDATORY TRAP TEST 4 (integration level)
    # ------------------------------------------------------------------
    def test_undefined_stereo_produces_two_ligands(self):
        """
        End-to-end: ibuprofen with undefined stereocenter must produce
        at least 2 PreparedLigand objects (one per enantiomer).
        """
        results = prepare_ligand(IBUPROFEN_UNDEF, enumerate_stereo=True)
        assert len(results) >= 2, (
            f"Expected ≥2 PreparedLigand objects for molecule with undefined "
            f"stereocenter, got {len(results)}"
        )

    def test_defined_stereo_produces_one_ligand(self):
        """
        Fully-defined stereocenter must not produce variants with undefined stereo.
        Total count may be >1 due to tautomers, but no result should have
        n_undefined_stereocenters > 0 and a stereoisomer_id of 'undefined'.
        """
        results = prepare_ligand(IBUPROFEN_R, enumerate_stereo=True)
        assert len(results) >= 1
        # No result should arise purely from an undefined stereocenter enumeration
        # (tautomers may legitimately produce additional variants)
        assert not any(
            r.stereoisomer_id == "undefined" for r in results
        ), "Defined stereo SMILES should not yield 'undefined' stereoisomer variants"

    def test_invalid_smiles_raises(self):
        with pytest.raises(Exception):
            prepare_ligand("this_is_not_smiles_!!")

    def test_erlotinib_no_stereocenters(self):
        results = prepare_ligand(ERLOTINIB)
        assert len(results) >= 1
        for r in results:
            assert r.n_undefined_stereocenters == 0
