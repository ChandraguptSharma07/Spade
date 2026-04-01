"""Tests for spade.core.structure."""

import io
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spade.core.structure import (
    AlphaFoldStructure,
    StructureParseError,
    _detect_pae_version,
    _parse_pae_json,
    fetch_structure,
    load_structure,
    write_structure,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# PAE JSON parsing — pure Python, no ProDy needed
# ---------------------------------------------------------------------------

class TestParsePaeJson:
    def test_v4_format(self):
        data = {"pae": [[0.1, 2.0], [2.0, 0.1]]}
        mat = _parse_pae_json(data)
        assert mat.shape == (2, 2)
        assert mat.dtype == np.float32
        assert mat[0, 1] == pytest.approx(2.0)

    def test_v2_format(self):
        data = {"predicted_aligned_error": [[0.1, 5.0], [5.0, 0.1]]}
        mat = _parse_pae_json(data)
        assert mat.shape == (2, 2)
        assert mat[0, 1] == pytest.approx(5.0)

    def test_list_wrapped_v2(self):
        data = [{"predicted_aligned_error": [[0.2, 3.0], [3.0, 0.2]]}]
        mat = _parse_pae_json(data)
        assert mat.shape == (2, 2)

    def test_unknown_format_raises(self):
        data = {"some_other_key": [[0, 1], [1, 0]], "another": True}
        with pytest.raises(StructureParseError) as exc_info:
            _parse_pae_json(data)
        # Error message must include the keys found
        assert "some_other_key" in str(exc_info.value)

    def test_non_square_raises(self):
        data = {"pae": [[0.1, 2.0, 3.0], [2.0, 0.1, 1.0]]}
        with pytest.raises(StructureParseError):
            _parse_pae_json(data)

    def test_list_with_multiple_entries_raises(self):
        data = [{"pae": [[0.1]]}, {"pae": [[0.2]]}]
        with pytest.raises(StructureParseError):
            _parse_pae_json(data)


class TestDetectPaeVersion:
    def test_v4(self):
        assert _detect_pae_version({"pae": []}) == "v4"

    def test_v2(self):
        assert _detect_pae_version({"predicted_aligned_error": []}) == "v2"

    def test_unknown(self):
        assert _detect_pae_version({"something": []}) == "unknown"


# ---------------------------------------------------------------------------
# load_structure — uses local fixtures
# ---------------------------------------------------------------------------

class TestLoadStructure:
    def test_loads_fixture(self):
        s = load_structure(
            str(FIXTURES / "egfr_kinase.pdb"),
            str(FIXTURES / "egfr_kinase_pae.json"),
        )
        assert isinstance(s, AlphaFoldStructure)
        assert s.n_residues == 4
        assert s.plddt.shape == (4,)
        assert s.pae_matrix.shape == (4, 4)
        assert s.af_version == "v4"

    def test_plddt_values_match_bfactors(self):
        s = load_structure(
            str(FIXTURES / "egfr_kinase.pdb"),
            str(FIXTURES / "egfr_kinase_pae.json"),
        )
        # Fixture B-factors: 92.5, 88.3, 65.1, 42.8
        expected = [92.5, 88.3, 65.1, 42.8]
        for i, exp in enumerate(expected):
            assert s.plddt[i] == pytest.approx(exp, abs=0.1)

    def test_v2_pae_format(self):
        s = load_structure(
            str(FIXTURES / "egfr_kinase.pdb"),
            str(FIXTURES / "egfr_kinase_pae_v2.json"),
        )
        assert s.af_version == "v2"
        assert s.pae_matrix.shape == (4, 4)

    def test_pae_matrix_is_float32(self):
        s = load_structure(
            str(FIXTURES / "egfr_kinase.pdb"),
            str(FIXTURES / "egfr_kinase_pae.json"),
        )
        assert s.pae_matrix.dtype == np.float32

    def test_missing_pdb_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_structure(
                str(tmp_path / "nonexistent.pdb"),
                str(FIXTURES / "egfr_kinase_pae.json"),
            )


# ---------------------------------------------------------------------------
# fetch_structure — mocked HTTP, no live network
# ---------------------------------------------------------------------------

def _make_mock_response(status_code: int, content: bytes = b"", json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = status_code == 200
    resp.content = content
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def _egfr_pdb_bytes() -> bytes:
    return (FIXTURES / "egfr_kinase.pdb").read_bytes()


def _egfr_pae_dict() -> dict:
    with open(FIXTURES / "egfr_kinase_pae.json") as f:
        return json.load(f)


class TestFetchStructure:
    def test_successful_fetch_v4(self):
        pdb_resp = _make_mock_response(200, _egfr_pdb_bytes())
        pae_resp = _make_mock_response(200, json_data=_egfr_pae_dict())

        with patch("spade.core.structure.requests.get") as mock_get:
            mock_get.side_effect = [pdb_resp, pae_resp]
            s = fetch_structure("P00533")

        assert s.uniprot_id == "P00533"
        assert s.af_version == "v4"
        assert s.n_residues == 4

    def test_404_raises_informative_error(self):
        not_found = _make_mock_response(404)

        with patch("spade.core.structure.requests.get") as mock_get:
            # All version attempts return 404
            mock_get.return_value = not_found
            with pytest.raises(StructureParseError) as exc_info:
                fetch_structure("BADID999")

        assert "BADID999" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()

    def test_falls_back_to_older_version(self):
        """If v4 returns 404, should try v3, v2, v1 before giving up."""
        not_found = _make_mock_response(404)
        pdb_resp = _make_mock_response(200, _egfr_pdb_bytes())
        pae_resp = _make_mock_response(200, json_data=_egfr_pae_dict())

        with patch("spade.core.structure.requests.get") as mock_get:
            # v4 PDB 404, then v3 PDB success
            mock_get.side_effect = [not_found, pdb_resp, pae_resp]
            s = fetch_structure("P00533")

        assert s.af_version == "v3"

    def test_unexpected_http_error_raises(self):
        error_resp = _make_mock_response(500)

        with patch("spade.core.structure.requests.get") as mock_get:
            mock_get.return_value = error_resp
            with pytest.raises(StructureParseError) as exc_info:
                fetch_structure("P00533")

        assert "500" in str(exc_info.value)

    def test_uniprot_id_is_uppercased(self):
        pdb_resp = _make_mock_response(200, _egfr_pdb_bytes())
        pae_resp = _make_mock_response(200, json_data=_egfr_pae_dict())

        with patch("spade.core.structure.requests.get") as mock_get:
            mock_get.side_effect = [pdb_resp, pae_resp]
            s = fetch_structure("p00533")  # lowercase input

        assert s.uniprot_id == "P00533"


# ---------------------------------------------------------------------------
# write_structure
# ---------------------------------------------------------------------------

class TestWriteStructure:
    def test_roundtrip(self, tmp_path):
        s = load_structure(
            str(FIXTURES / "egfr_kinase.pdb"),
            str(FIXTURES / "egfr_kinase_pae.json"),
        )
        out = tmp_path / "out.pdb"
        write_structure(s.atoms, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_atom_count_preserved(self, tmp_path):
        import prody
        s = load_structure(
            str(FIXTURES / "egfr_kinase.pdb"),
            str(FIXTURES / "egfr_kinase_pae.json"),
        )
        n_atoms_before = s.atoms.numAtoms()
        out = tmp_path / "out.pdb"
        write_structure(s.atoms, str(out))

        reloaded = prody.parsePDB(str(out))
        assert reloaded.numAtoms() == n_atoms_before
