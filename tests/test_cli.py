"""Tests for spade.interfaces.cli."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from spade.interfaces.cli import app

runner = CliRunner()


class TestCliHelp:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "SPADE" in result.output

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--ligand" in result.output
        assert "--uniprot" in result.output

    def test_prep_help(self):
        result = runner.invoke(app, ["prep", "--help"])
        assert result.exit_code == 0
        assert "--uniprot" in result.output

    def test_dock_help(self):
        result = runner.invoke(app, ["dock", "--help"])
        assert result.exit_code == 0
        assert "--ensemble-dir" in result.output

    def test_interactive_help(self):
        result = runner.invoke(app, ["interactive", "--help"])
        assert result.exit_code == 0


class TestCliRun:
    def test_missing_ligand_exits_nonzero(self):
        # --ligand is required
        result = runner.invoke(app, ["run", "--uniprot", "P00533"])
        assert result.exit_code != 0

    def test_missing_structure_exits_nonzero(self):
        # neither --uniprot nor --pdb provided
        result = runner.invoke(app, ["run", "--ligand", "CC"])
        assert result.exit_code != 0

    def test_run_with_local_pdb(self, tmp_path):
        """Run command with a local PDB + PAE file should reach dock_ensemble."""
        from pathlib import Path

        pdb_fixture = Path(__file__).parent / "fixtures" / "egfr_kinase.pdb"
        pae_fixture = Path(__file__).parent / "fixtures" / "egfr_kinase_pae.json"
        out = tmp_path / "out"

        # Mock the expensive parts
        mock_conformers = [MagicMock()]
        mock_conformers[0].getData.return_value = [0.3]
        mock_conformers[0].numAtoms.return_value = 10

        from spade.core.docking import BoundingBox, DockingResult
        import numpy as np

        mock_dr = DockingResult(
            conformer_index=0, conformer_ca_rmsd=0.3, poses=[],
            bounding_box=BoundingBox(np.zeros(3), np.full(3, 20.0)),
            docking_time_seconds=0.5,
        )

        from spade.core.clustering import ConsensusResult, PoseCluster, PoseResult
        mock_pose = PoseResult(0, -9.0, np.zeros((5, 3), np.float32), 0)
        mock_cluster = PoseCluster(
            cluster_id=0, representative_pose=mock_pose, member_poses=[mock_pose],
            n_conformers_represented=1, fraction_ensemble=0.5,
            mean_score=-9.0, score_std=0.5, consensus_score=-9.0,
            interaction_fingerprint=np.zeros(64, np.float32),
        )
        mock_consensus = ConsensusResult(
            top_cluster=mock_cluster, all_clusters=[mock_cluster],
            pose_confidence=0.8, site_confidence="high",
            n_total_poses=1, n_clusters=1,
        )

        with patch("spade.core.ensemble.EnsembleGenerator") as MockGen, \
             patch("spade.core.docking.dock_ensemble", return_value=[mock_dr]) as _mock_de, \
             patch("spade.core.clustering.cluster_poses", return_value=mock_consensus) as _mock_cp:
            mock_gen_inst = MockGen.return_value
            mock_gen_inst.generate.return_value = mock_conformers

            result = runner.invoke(app, [
                "run",
                "--pdb", str(pdb_fixture),
                "--pae", str(pae_fixture),
                "--ligand", "CC(=O)Oc1ccccc1C(=O)O",
                "--conformers", "2",
                "--output", str(out),
            ])

        assert result.exit_code == 0, f"CLI exited with {result.exit_code}:\n{result.output}"
        assert out.exists()
        assert (out / "provenance.json").exists()
        assert (out / "report.html").exists()


class TestCliPrep:
    def test_prep_requires_uniprot(self):
        result = runner.invoke(app, ["prep"])
        assert result.exit_code != 0

    def test_prep_calls_fetch_and_write(self, tmp_path):
        import numpy as np

        mock_structure = MagicMock()
        mock_structure.n_residues = 100
        mock_structure.af_version = "v4"
        mock_structure.plddt = np.full(100, 85.0)
        mock_structure.atoms = MagicMock()

        out = tmp_path / "prep_out"

        with patch("spade.core.structure.fetch_structure", return_value=mock_structure), \
             patch("spade.core.structure.write_structure") as mock_write:
            result = runner.invoke(app, ["prep", "--uniprot", "P12345", "--output", str(out)])

        assert result.exit_code == 0
        mock_write.assert_called_once()


class TestCliDock:
    def test_dock_missing_ensemble_dir_exits(self, tmp_path):
        result = runner.invoke(app, [
            "dock",
            "--ensemble-dir", str(tmp_path / "nonexistent"),
            "--ligand", "CC",
        ])
        assert result.exit_code != 0

    def test_dock_empty_dir_exits(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = runner.invoke(app, [
            "dock",
            "--ensemble-dir", str(empty),
            "--ligand", "CC",
        ])
        assert result.exit_code != 0
