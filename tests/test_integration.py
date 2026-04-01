"""
Integration test — full SPADE pipeline smoke test.

Uses local fixture files (no live network required).
AutoDock Vina is mocked so the test runs in CI without the binary.

Assertions:
  - Pipeline completes without error
  - Erlotinib consensus score < decoy consensus score (lower = better in Vina)
  - provenance.json contains all required fields
  - report.html is valid HTML (contains DOCTYPE and embedded provenance JSON)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import prody

prody.confProDy(verbosity="none")

from spade.core.clustering import cluster_poses
from spade.core.docking import BoundingBox, DockingResult, PoseResult
from spade.core.flexibility import build_flexibility_profile
from spade.core.ensemble import EnsembleGenerator
from spade.core.ligand import prepare_ligand
from spade.core.report import ConformerSummary, RunProvenance, generate_report
from spade.core.structure import load_structure

FIXTURES = Path(__file__).parent / "fixtures"

ERLOTINIB_SMILES = "CCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOCCO"
DECOY_SMILES = "CCCCCC"  # n-hexane — should score much worse than erlotinib

REQUIRED_PROVENANCE_FIELDS = {
    "run_id",
    "timestamp",
    "spade_version",
    "uniprot_id",
    "ligand_smiles",
    "n_conformers_generated",
    "n_total_poses",
    "n_clusters",
    "top_cluster_score",
    "site_confidence",
    "plddt_mean",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_egfr():
    return load_structure(
        str(FIXTURES / "egfr_kinase.pdb"),
        str(FIXTURES / "egfr_kinase_pae.json"),
    )


def _make_pocket(structure):
    return np.array([0, 1, 2])


def _make_mock_docking_results(conformers, ligands, base_score: float):
    """
    Return synthetic DockingResult objects with a known base_score.
    Used to avoid Vina dependency in integration tests.
    """
    results = []
    bb = BoundingBox(center=np.zeros(3), size=np.full(3, 20.0))
    for ci, conf in enumerate(conformers):
        for lig in ligands:
            # Poses near the base score with small random spread
            rng = np.random.RandomState(ci + int(abs(base_score) * 10))
            poses = [
                PoseResult(
                    pose_index=pi,
                    score_kcal_mol=base_score + rng.uniform(-0.5, 0.5),
                    coordinates=rng.randn(10, 3).astype(np.float32),
                    conformer_index=ci,
                )
                for pi in range(3)
            ]
            results.append(DockingResult(
                conformer_index=ci,
                conformer_ca_rmsd=float(conf.getData("ca_rmsd_from_ref")[0])
                if conf.getData("ca_rmsd_from_ref") is not None else 0.0,
                poses=poses,
                bounding_box=bb,
                docking_time_seconds=0.1,
            ))
    return results


# ---------------------------------------------------------------------------
# Full pipeline smoke test
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def setup_method(self):
        self.structure = _load_egfr()
        self.pocket = _make_pocket(self.structure)
        ca_sel = self.structure.atoms.select("name CA")
        ca_coords = ca_sel.getCoords()
        self.profile = build_flexibility_profile(
            self.structure.plddt,
            self.structure.pae_matrix,
            self.pocket,
            ca_coords,
        )

    def test_pipeline_completes_without_error(self):
        gen = EnsembleGenerator(self.structure, self.profile, n_conformers=3)
        conformers = gen.generate()
        assert len(conformers) > 0

        ligands = prepare_ligand(ERLOTINIB_SMILES)
        assert len(ligands) > 0

        docking_results = _make_mock_docking_results(conformers, ligands, base_score=-9.5)
        consensus = cluster_poses(docking_results, conformers, ligands[0].mol)

        assert consensus is not None
        assert consensus.n_total_poses > 0

    def test_erlotinib_scores_better_than_decoy(self):
        """
        Erlotinib consensus score must be lower (better) than a simple decoy.
        Uses mock scores: erlotinib at -9.5 kcal/mol, decoy at -3.0 kcal/mol.
        """
        gen = EnsembleGenerator(self.structure, self.profile, n_conformers=3)
        conformers = gen.generate()

        erlotinib_ligands = prepare_ligand(ERLOTINIB_SMILES)
        decoy_ligands = prepare_ligand(DECOY_SMILES)

        # Mock docking: erlotinib binds much better than hexane
        erl_results = _make_mock_docking_results(conformers, erlotinib_ligands, base_score=-9.5)
        decoy_results = _make_mock_docking_results(conformers, decoy_ligands, base_score=-3.0)

        erl_consensus = cluster_poses(erl_results, conformers, erlotinib_ligands[0].mol)
        decoy_consensus = cluster_poses(decoy_results, conformers, decoy_ligands[0].mol)

        erl_score = erl_consensus.top_cluster.mean_score
        decoy_score = decoy_consensus.top_cluster.mean_score

        assert erl_score < decoy_score, (
            f"Erlotinib ({erl_score:.2f}) should score better (lower) than decoy ({decoy_score:.2f})"
        )

    def test_provenance_json_completeness(self, tmp_path):
        gen = EnsembleGenerator(self.structure, self.profile, n_conformers=2)
        conformers = gen.generate()
        ligands = prepare_ligand(ERLOTINIB_SMILES)
        docking_results = _make_mock_docking_results(conformers, ligands, base_score=-9.5)
        consensus = cluster_poses(docking_results, conformers, ligands[0].mol)

        conf_summaries = [
            ConformerSummary(
                conformer_index=dr.conformer_index,
                ca_rmsd_from_ref=dr.conformer_ca_rmsd,
                n_poses=len(dr.poses),
                best_score_kcal_mol=min(p.score_kcal_mol for p in dr.poses) if dr.poses else 0.0,
                docking_time_seconds=dr.docking_time_seconds,
            )
            for dr in docking_results
        ]

        prov = RunProvenance(
            uniprot_id="P00533",
            af_version=self.structure.af_version,
            n_residues=self.structure.n_residues,
            pocket_residues=self.pocket.tolist(),
            ligand_smiles=ERLOTINIB_SMILES,
            n_ligand_variants=len(ligands),
            n_conformers_requested=2,
            n_conformers_generated=len(conformers),
            n_total_poses=consensus.n_total_poses,
            n_clusters=consensus.n_clusters,
            top_cluster_score=consensus.top_cluster.mean_score,
            top_cluster_fraction_ensemble=consensus.top_cluster.fraction_ensemble,
            site_confidence=consensus.site_confidence,
            conformer_summaries=conf_summaries,
            plddt_mean=float(self.structure.plddt.mean()),
            plddt_std=float(self.structure.plddt.std()),
        )

        generate_report(prov, str(tmp_path))

        prov_path = tmp_path / "provenance.json"
        assert prov_path.exists(), "provenance.json must be created"

        with open(prov_path) as f:
            pj = json.load(f)

        missing = REQUIRED_PROVENANCE_FIELDS - set(pj.keys())
        assert not missing, f"provenance.json is missing fields: {missing}"

        assert pj["uniprot_id"] == "P00533"
        assert pj["ligand_smiles"] == ERLOTINIB_SMILES
        assert len(pj["run_id"]) == 36  # UUID format

    def test_report_html_is_valid(self, tmp_path):
        gen = EnsembleGenerator(self.structure, self.profile, n_conformers=2)
        conformers = gen.generate()
        ligands = prepare_ligand(ERLOTINIB_SMILES)
        docking_results = _make_mock_docking_results(conformers, ligands, base_score=-9.5)
        consensus = cluster_poses(docking_results, conformers, ligands[0].mol)

        prov = RunProvenance(
            uniprot_id="P00533",
            ligand_smiles=ERLOTINIB_SMILES,
            n_conformers_generated=len(conformers),
            n_total_poses=consensus.n_total_poses,
            n_clusters=consensus.n_clusters,
            top_cluster_score=consensus.top_cluster.mean_score,
            site_confidence=consensus.site_confidence,
            plddt_mean=float(self.structure.plddt.mean()),
            plddt_std=float(self.structure.plddt.std()),
        )

        generate_report(prov, str(tmp_path))

        html_path = tmp_path / "report.html"
        assert html_path.exists(), "report.html must be created"

        html = html_path.read_text(encoding="utf-8")

        # Must be valid enough HTML
        assert "<!DOCTYPE html>" in html or "<!doctype html>" in html.lower(), (
            "report.html must start with DOCTYPE declaration"
        )
        assert "<html" in html.lower(), "report.html must contain <html> tag"
        assert "</html>" in html.lower(), "report.html must close </html>"

        # Embedded provenance JSON
        assert '<script type="application/json"' in html, (
            "report.html must embed provenance JSON in <script type='application/json'>"
        )

        # Content sanity
        assert "P00533" in html
        assert ERLOTINIB_SMILES in html or "erlotinib" in html.lower() or "P00533" in html


# ---------------------------------------------------------------------------
# Individual module contracts (lighter smoke tests)
# ---------------------------------------------------------------------------

class TestModuleContracts:
    def test_structure_loads_without_network(self):
        structure = _load_egfr()
        assert structure.n_residues > 0
        assert structure.pae_matrix.shape[0] == structure.n_residues
        assert len(structure.plddt) == structure.n_residues

    def test_flexibility_profile_pocket_local(self):
        structure = _load_egfr()
        pocket = np.array([0, 1])
        ca_coords = structure.atoms.select("name CA").getCoords()
        profile = build_flexibility_profile(
            structure.plddt, structure.pae_matrix, pocket, ca_coords
        )
        assert profile.pocket_residues is not None
        assert profile.mode_weight_vector is not None

    def test_ensemble_generator_returns_atom_groups(self):
        structure = _load_egfr()
        pocket = np.array([0, 1])
        ca_coords = structure.atoms.select("name CA").getCoords()
        profile = build_flexibility_profile(
            structure.plddt, structure.pae_matrix, pocket, ca_coords
        )
        gen = EnsembleGenerator(structure, profile, n_conformers=2)
        conformers = gen.generate()
        for conf in conformers:
            assert isinstance(conf, prody.AtomGroup)
            assert conf.numAtoms() == structure.atoms.numAtoms()

    def test_ligand_preparation_erlotinib(self):
        ligands = prepare_ligand(ERLOTINIB_SMILES)
        assert len(ligands) >= 1
        for lig in ligands:
            assert len(lig.pdbqt_string) > 0
            assert lig.n_undefined_stereocenters == 0

    def test_clustering_with_known_scores(self):
        """Cluster two groups of poses with clearly different scores."""
        bb = BoundingBox(center=np.zeros(3), size=np.full(3, 20.0))
        good_poses = [
            PoseResult(i, -9.0, np.ones((5, 3), np.float32) * i, 0)
            for i in range(3)
        ]
        bad_poses = [
            PoseResult(i, -3.0, np.ones((5, 3), np.float32) * (i + 10), 1)
            for i in range(3)
        ]
        dr_good = DockingResult(0, 0.3, good_poses, bb, 1.0)
        dr_bad = DockingResult(1, 0.7, bad_poses, bb, 1.0)
        consensus = cluster_poses([dr_good, dr_bad], conformers=[], ligand_mol=None)
        assert consensus.top_cluster.mean_score < -5.0, (
            "Top cluster should contain the good-scoring poses"
        )
