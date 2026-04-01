"""
structure.py — AlphaFold2 structure fetching and parsing.

Fetches structures from the EBI AlphaFold API and parses PAE JSON.
The canonical internal representation is a ProDy AtomGroup with pLDDT
scores stored in the B-factor field.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import numpy as np
import requests

try:
    import prody
    prody.confProDy(verbosity="none")
except ImportError as e:
    raise ImportError("ProDy is required: conda install -c conda-forge prody") from e


EBI_BASE = "https://alphafold.ebi.ac.uk/files"
_MODEL_VERSIONS = ["v4", "v3", "v2", "v1"]


class StructureParseError(Exception):
    """Raised when a structure or PAE file cannot be parsed."""


@dataclass
class AlphaFoldStructure:
    """
    Container for a fetched/loaded AlphaFold2 structure.

    atoms       — full ProDy AtomGroup; pLDDT stored in the B-factor field
    pae_matrix  — (N_res, N_res) float32 array of predicted aligned errors (Angstrom)
    plddt       — (N_res,) float32 per-residue confidence scores
    uniprot_id  — UniProt accession used to retrieve this structure
    af_version  — model version string detected from PAE JSON, e.g. 'v4'
    n_residues  — number of residues (= len(plddt))
    """

    atoms: "prody.AtomGroup"
    pae_matrix: np.ndarray
    plddt: np.ndarray
    uniprot_id: str
    af_version: str
    n_residues: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_structure(uniprot_id: str) -> AlphaFoldStructure:
    """
    Download an AlphaFold2 structure and its PAE JSON from the EBI API.

    Tries model versions v4 → v1 in order until one is found.
    Raises StructureParseError if no version is available or the UniProt ID
    does not exist.
    """
    uniprot_id = uniprot_id.strip().upper()

    pdb_bytes = None
    pae_data = None
    found_version = None

    for version in _MODEL_VERSIONS:
        pdb_url = f"{EBI_BASE}/AF-{uniprot_id}-F1-model_{version}.pdb"
        pae_url = f"{EBI_BASE}/AF-{uniprot_id}-F1-predicted_aligned_error_{version}.json"

        pdb_resp = requests.get(pdb_url, timeout=30)
        if pdb_resp.status_code == 404:
            continue
        if not pdb_resp.ok:
            raise StructureParseError(
                f"Unexpected HTTP {pdb_resp.status_code} fetching PDB for {uniprot_id}"
            )

        pae_resp = requests.get(pae_url, timeout=30)
        if not pae_resp.ok:
            raise StructureParseError(
                f"PDB found but PAE JSON unavailable (HTTP {pae_resp.status_code}) "
                f"for {uniprot_id} {version}"
            )

        pdb_bytes = pdb_resp.content
        pae_data = pae_resp.json()
        found_version = version
        break

    if pdb_bytes is None:
        raise StructureParseError(
            f"UniProt ID '{uniprot_id}' not found in AlphaFold EBI database. "
            "Check the accession and ensure it has an AF2 model."
        )

    ag = _parse_pdb_bytes(pdb_bytes, uniprot_id)
    pae_matrix = _parse_pae_json(pae_data)
    plddt = _extract_plddt(ag)

    return AlphaFoldStructure(
        atoms=ag,
        pae_matrix=pae_matrix,
        plddt=plddt,
        uniprot_id=uniprot_id,
        af_version=found_version,
        n_residues=len(plddt),
    )


def load_structure(pdb_path: str, pae_path: str) -> AlphaFoldStructure:
    """
    Load an AlphaFold2 structure from local PDB and PAE JSON files.
    Useful for offline work and testing with cached fixtures.
    """
    ag = prody.parsePDB(pdb_path)
    if ag is None:
        raise StructureParseError(f"ProDy could not parse PDB file: {pdb_path}")

    with open(pae_path, "r") as fh:
        pae_data = json.load(fh)

    pae_matrix = _parse_pae_json(pae_data)
    plddt = _extract_plddt(ag)

    # Detect version from JSON keys
    af_version = _detect_pae_version(pae_data)

    # Extract UniProt ID from the PDB TITLE/REMARK if present, else use filename stem
    uniprot_id = _guess_uniprot_id(ag, pdb_path)

    return AlphaFoldStructure(
        atoms=ag,
        pae_matrix=pae_matrix,
        plddt=plddt,
        uniprot_id=uniprot_id,
        af_version=af_version,
        n_residues=len(plddt),
    )


def write_structure(ag: "prody.AtomGroup", path: str) -> None:
    """
    Write a ProDy AtomGroup to a PDB file using ProDy's writer.

    Always go through ProDy's writer — never construct PDB strings manually.
    This enforces standard IUPAC atom naming and ordering, which is required
    for correct behaviour in ProLIF and Vina.
    """
    prody.writePDB(path, ag)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_pdb_bytes(pdb_bytes: bytes, uniprot_id: str) -> "prody.AtomGroup":
    """Parse PDB bytes into a ProDy AtomGroup via an in-memory stream."""
    pdb_str = pdb_bytes.decode("utf-8", errors="replace")
    ag = prody.parsePDBStream(io.StringIO(pdb_str))
    if ag is None:
        raise StructureParseError(
            f"ProDy failed to parse PDB content for {uniprot_id}. "
            "The file may be malformed or empty."
        )
    ag.setTitle(uniprot_id)
    return ag


def _detect_pae_version(data: dict) -> str:
    """Return a version string based on which keys are present in the PAE JSON."""
    if "predicted_aligned_error" in data:
        return "v2"
    if "pae" in data:
        return "v4"
    # Some intermediate versions wrap the matrix in a list of dicts
    if isinstance(data, list) and len(data) > 0 and "predicted_aligned_error" in data[0]:
        return "v2_list"
    return "unknown"


def _parse_pae_json(data: dict | list) -> np.ndarray:
    """
    Parse PAE JSON into an (N_res, N_res) float32 numpy array.

    Handles known EBI AlphaFold PAE formats:
      - v1/v2: {"predicted_aligned_error": [[...]]}
      - v3/v4: {"pae": [[...]]}
      - list wrapper: [{"predicted_aligned_error": [[...]]}]  (some API versions)

    Raises StructureParseError with the keys found if the format is unrecognised.
    """
    # Unwrap list-of-one-dict format
    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], dict):
            data = data[0]
        else:
            raise StructureParseError(
                f"PAE JSON is a list with {len(data)} elements; expected a single dict or "
                "a list containing one dict."
            )

    if "predicted_aligned_error" in data:
        raw = data["predicted_aligned_error"]
    elif "pae" in data:
        raw = data["pae"]
    else:
        raise StructureParseError(
            f"Unknown PAE JSON format. Keys found: {list(data.keys())}. "
            "Expected 'predicted_aligned_error' (v1/v2) or 'pae' (v3/v4)."
        )

    # NOTE on indexing: PAE[i, j] = error when residue j is used as reference to
    # predict residue i's position. Row = aligned residue, col = reference residue.
    # EBI JSON is stored row-major: outer list = rows (aligned), inner = cols (reference).
    # np.array() preserves this ordering correctly — no transpose needed.
    matrix = np.array(raw, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise StructureParseError(
            f"PAE matrix has unexpected shape {matrix.shape}; expected a square 2D array."
        )
    return matrix


def _extract_plddt(ag: "prody.AtomGroup") -> np.ndarray:
    """
    Extract per-residue pLDDT scores from the B-factor field of a ProDy AtomGroup.

    AlphaFold2 PDB files store per-atom pLDDT in the B-factor column.
    We take one value per residue (the Cα B-factor, or first atom if Cα not found).
    Returns float32 array of shape (N_res,).
    """
    ca_sel = ag.select("name CA")
    if ca_sel is None or len(ca_sel) == 0:
        # Fallback: one B-factor per residue from any atom
        bfactors = []
        for res in ag.iterResidues():
            bfactors.append(float(res.getBetas()[0]))
        return np.array(bfactors, dtype=np.float32)

    return ca_sel.getBetas().astype(np.float32)


def _guess_uniprot_id(ag: "prody.AtomGroup", pdb_path: str) -> str:
    """
    Try to read the UniProt ID from the AtomGroup title, else fall back to
    the PDB filename stem.
    """
    title = ag.getTitle()
    if title and len(title) >= 6:
        # AF PDB titles are often the full accession
        parts = title.strip().split()
        for part in parts:
            if 6 <= len(part) <= 10 and part.isalnum():
                return part.upper()

    from pathlib import Path
    stem = Path(pdb_path).stem
    # Strip common suffixes like _model_v4, _kinase, etc.
    candidate = stem.split("_")[0]
    return candidate.upper()
