"""
ligand.py — Ligand preparation: tautomers, protomers, stereoisomers, 3D, PDBQT.

Scientific correctness requirement: if a SMILES string contains undefined
stereocenters, ALL stereoisomers are enumerated and docked separately.
A chiral drug's enantiomers can have opposite pharmacological profiles.
Silently picking one enantiomer is not acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.EnumerateStereoisomers import (
        EnumerateStereoisomers,
        StereoEnumerationOptions,
    )
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError as e:
    raise ImportError(
        "RDKit is required: conda install -c conda-forge rdkit"
    ) from e

try:
    from meeko import MoleculePreparation
except ImportError as e:
    raise ImportError("Meeko is required: pip install meeko") from e

_DIMORPHITE_AVAILABLE = False
try:
    import dimorphite_dl
    _DIMORPHITE_AVAILABLE = True
except ImportError:
    pass


@dataclass
class PreparedLigand:
    """
    A single prepared ligand variant ready for docking.

    One input SMILES may produce multiple PreparedLigand objects when:
    - stereocenters are undefined (one per stereoisomer)
    - multiple tautomers are requested (one per tautomer)
    - multiple protomers at target pH (one per protonation state)
    """

    mol: Chem.Mol
    pdbqt_string: str
    smiles_input: str
    stereoisomer_id: str              # e.g. 'R,S' or 'undefined' or 'flat'
    n_undefined_stereocenters: int    # 0 if all stereocenters are defined
    tautomer_id: int                  # 0-based index in the tautomer enumeration
    protomer_ph: float                # pH used for protonation state


def prepare_ligand(
    smiles: str,
    ph: float = 7.4,
    enumerate_stereo: bool = True,
    n_conformers: int = 10,
) -> list[PreparedLigand]:
    """
    Prepare a ligand from a SMILES string for ensemble docking.

    Steps:
    1. Parse and validate SMILES
    2. Enumerate tautomers (top 3 by RDKit TautomerEnumerator)
    3. Enumerate protomers at target pH (Dimorphite-DL if available)
    4. Detect undefined stereocenters
    5. Enumerate all stereoisomers if stereocenters are undefined
    6. Generate 3D conformers with ETKDGv3 + MMFF optimisation
    7. Prepare Meeko PDBQT for each variant

    Returns a list of PreparedLigand objects — may contain >1 entry when
    stereocenters are undefined or multiple tautomers/protomers are generated.
    """
    mol = _parse_smiles(smiles)

    tautomers = _enumerate_tautomers(mol, max_tautomers=3)
    results: list[PreparedLigand] = []

    for tau_id, tau_mol in enumerate(tautomers):
        protomers = _enumerate_protomers(tau_mol, ph)
        for proto_mol in protomers:
            stereo_variants = _enumerate_stereoisomers(proto_mol, enumerate_stereo)
            for stereo_mol, stereo_id, n_undef in stereo_variants:
                mol_3d = _generate_3d(stereo_mol, n_conformers)
                if mol_3d is None:
                    continue
                pdbqt = _prepare_pdbqt(mol_3d)
                if pdbqt is None:
                    continue
                results.append(PreparedLigand(
                    mol=mol_3d,
                    pdbqt_string=pdbqt,
                    smiles_input=smiles,
                    stereoisomer_id=stereo_id,
                    n_undefined_stereocenters=n_undef,
                    tautomer_id=tau_id,
                    protomer_ph=ph,
                ))

    if not results:
        raise ValueError(
            f"Could not prepare any ligand variant from SMILES: {smiles!r}. "
            "Check that the SMILES is valid and 3D conformer generation succeeded."
        )

    return results


# ---------------------------------------------------------------------------
# Internal steps
# ---------------------------------------------------------------------------

def _parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def _enumerate_tautomers(mol: Chem.Mol, max_tautomers: int = 3) -> list[Chem.Mol]:
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(max_tautomers)
    try:
        tautomers = list(enumerator.Enumerate(mol))
    except Exception:
        tautomers = [mol]

    if not tautomers:
        tautomers = [mol]

    return tautomers[:max_tautomers]


def _enumerate_protomers(mol: Chem.Mol, ph: float) -> list[Chem.Mol]:
    """
    Enumerate protonation states at the target pH.
    Uses Dimorphite-DL if available, else returns the input molecule unchanged.
    """
    if not _DIMORPHITE_AVAILABLE:
        return [mol]

    try:
        protonated = dimorphite_dl.run_with_mol_list(
            [mol],
            min_ph=ph - 0.5,
            max_ph=ph + 0.5,
            pka_precision=0.5,
            silent=True,
        )
        if not protonated:
            return [mol]
        return protonated
    except Exception:
        return [mol]


def _enumerate_stereoisomers(
    mol: Chem.Mol,
    enumerate: bool,
) -> list[tuple[Chem.Mol, str, int]]:
    """
    Detect undefined stereocenters and optionally enumerate all stereoisomers.

    Returns list of (mol, stereoisomer_id, n_undefined_stereocenters).

    If enumerate=False or all stereocenters are defined, returns a single entry.
    """
    # Find all chiral centers, including unassigned ones
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    n_undefined = sum(1 for _, chirality in chiral_centers if chirality == "?")

    if n_undefined == 0 or not enumerate:
        stereo_id = _stereo_label(chiral_centers)
        return [(mol, stereo_id, n_undefined)]

    # Enumerate all stereoisomers
    opts = StereoEnumerationOptions(unique=True, onlyUnassigned=True)
    isomers = list(EnumerateStereoisomers(mol, options=opts))
    if not isomers:
        return [(mol, "undefined", n_undefined)]

    result = []
    for isomer in isomers:
        centers = Chem.FindMolChiralCenters(isomer, includeUnassigned=True)
        label = _stereo_label(centers)
        result.append((isomer, label, 0))  # after enumeration, all centers are defined
    return result


def _stereo_label(chiral_centers: list) -> str:
    """Generate a human-readable stereoisomer ID from FindMolChiralCenters output."""
    if not chiral_centers:
        return "flat"
    parts = []
    for atom_idx, chirality in chiral_centers:
        parts.append(chirality if chirality != "?" else "undefined")
    return ",".join(parts)


def _generate_3d(mol: Chem.Mol, n_conformers: int = 10) -> Optional[Chem.Mol]:
    """
    Generate a 3D conformer using ETKDGv3 + MMFF optimisation.
    Returns None if embedding fails.
    """
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1

    # EmbedMultipleConfs for flexibility; we use the first conformer for docking prep
    ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_conformers, params=params)
    result = 0 if ids else -1
    if result == -1:
        # Fallback: single conformer with ETKDG
        result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
    if result == -1:
        return None

    try:
        AllChem.MMFFOptimizeMolecule(mol_h)
    except Exception:
        pass  # optimisation failure is non-fatal; use the unoptimised geometry

    return mol_h


def _prepare_pdbqt(mol: Chem.Mol) -> Optional[str]:
    """
    Prepare a PDBQT string using Meeko.
    Returns None if Meeko fails (e.g. unsupported element).
    """
    try:
        from meeko import PDBQTWriterLegacy
        prep = MoleculePreparation()
        mol_setups = prep.prepare(mol)
        if not mol_setups:
            return None
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
        if not is_ok:
            return None
        return pdbqt_string
    except Exception:
        # Fall back to legacy API for older Meeko versions
        try:
            prep = MoleculePreparation()
            prep.prepare(mol)
            return prep.write_pdbqt_string()  # type: ignore[attr-defined]
        except Exception:
            return None
