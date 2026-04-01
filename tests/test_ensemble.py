"""Tests for spade.core.ensemble."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import prody

prody.confProDy(verbosity="none")

from spade.core.ensemble import (
    MAX_CA_RMSD_ANGSTROM,
    EnsembleGenerator,
    _residue_indices_to_resnums,
)
from spade.core.flexibility import build_flexibility_profile
from spade.core.structure import load_structure

FIXTURES = Path(__file__).parent / "fixtures"


def _load_test_structure():
    return load_structure(
        str(FIXTURES / "egfr_kinase.pdb"),
        str(FIXTURES / "egfr_kinase_pae.json"),
    )


def _make_profile(structure, pocket_residues=None):
    if pocket_residues is None:
        pocket_residues = np.array([1, 2])
    ca_sel = structure.atoms.select("name CA")
    ca_coords = ca_sel.getCoords()
    return build_flexibility_profile(
        structure.plddt,
        structure.pae_matrix,
        pocket_residues,
        ca_coords,
    )


class TestEnsembleGenerator:
    def setup_method(self):
        self.structure = _load_test_structure()
        self.profile = _make_profile(self.structure)

    def test_generate_returns_list(self):
        gen = EnsembleGenerator(self.structure, self.profile, n_conformers=3)
        conformers = gen.generate()
        assert isinstance(conformers, list)
        assert len(conformers) > 0

    def test_all_conformers_are_atom_groups(self):
        gen = EnsembleGenerator(self.structure, self.profile, n_conformers=3)
        conformers = gen.generate()
        for conf in conformers:
            assert isinstance(conf, prody.AtomGroup), "Each conformer must be a ProDy AtomGroup"

    def test_atom_count_preserved(self):
        """Every conformer must have the same atom count as the reference structure."""
        gen = EnsembleGenerator(self.structure, self.profile, n_conformers=3)
        conformers = gen.generate()
        n_ref = self.structure.atoms.numAtoms()
        for conf in conformers:
            assert conf.numAtoms() == n_ref

    def test_ca_rmsd_cap_enforced(self):
        """No conformer may exceed MAX_CA_RMSD_ANGSTROM from the reference."""
        gen = EnsembleGenerator(self.structure, self.profile, n_conformers=5)
        conformers = gen.generate()
        for conf in conformers:
            rmsd = gen._ca_rmsd(self.structure.atoms, conf)
            assert rmsd <= MAX_CA_RMSD_ANGSTROM + 1e-6, (
                f"Conformer Cα RMSD {rmsd:.3f} Å exceeds cap {MAX_CA_RMSD_ANGSTROM} Å"
            )

    def test_provenance_attributes_attached(self):
        # Provenance stored via setData (AtomGroup is Cython — no __dict__)
        gen = EnsembleGenerator(self.structure, self.profile, n_conformers=2)
        conformers = gen.generate()
        for conf in conformers:
            assert conf.getData("conformer_index") is not None
            assert conf.getData("ca_rmsd_from_ref") is not None
            assert int(conf.getData("conformer_index")[0]) >= 0
            assert float(conf.getData("ca_rmsd_from_ref")[0]) >= 0.0

    def test_conformer_indices_are_sequential(self):
        gen = EnsembleGenerator(self.structure, self.profile, n_conformers=3)
        conformers = gen.generate()
        for i, conf in enumerate(conformers):
            assert int(conf.getData("conformer_index")[0]) == i

    def test_repacker_called_on_clashing_conformers(self):
        """If detect_clashes returns a non-empty list, repacker.repack must be called."""
        mock_repacker = MagicMock()
        mock_repacker.repack.side_effect = lambda ag, clashing: ag.copy()

        with patch(
            "spade.core.ensemble.DunbrackRepacker._detect_clashes",
            return_value=[0, 1],  # pretend all conformers have clashes
        ):
            gen = EnsembleGenerator(
                self.structure, self.profile, n_conformers=2, repacker=mock_repacker
            )
            gen.generate()

        assert mock_repacker.repack.called, (
            "repacker.repack should have been called for conformers with clashes"
        )

    def test_n_conformers_respected(self):
        gen = EnsembleGenerator(self.structure, self.profile, n_conformers=2)
        conformers = gen.generate()
        assert len(conformers) <= 2

    def test_ca_rmsd_zero_for_identical(self):
        gen = EnsembleGenerator(self.structure, self.profile, n_conformers=1)
        rmsd = gen._ca_rmsd(self.structure.atoms, self.structure.atoms.copy())
        assert rmsd == pytest.approx(0.0, abs=1e-5)


class TestResidueIndicesToResnums:
    def test_basic(self):
        ag = prody.parsePDB(str(FIXTURES / "egfr_kinase.pdb"))
        resnums = _residue_indices_to_resnums(ag, np.array([0, 1, 2]))
        assert len(resnums) == 3
        assert all(isinstance(r, int) for r in resnums)

    def test_out_of_range_skipped(self):
        ag = prody.parsePDB(str(FIXTURES / "egfr_kinase.pdb"))
        resnums = _residue_indices_to_resnums(ag, np.array([0, 9999]))
        assert len(resnums) == 1  # only valid index
