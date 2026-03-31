# SPADE: Structural PAE-Aware Docking Ensembles

**WIP** — ensemble docking pipeline that actually uses AlphaFold2's confidence outputs instead of throwing them away.

Concept: fetch AF2 structure → classify residues by pLDDT → build PAE-weighted flexibility graph (pocket-local only) → NMA ensemble → dock each conformer → cluster by PLIF → consensus score + uncertainty.

More details once the core modules are in shape.

## Install

```bash
conda create -n spade python=3.10
conda activate spade
conda install -c conda-forge rdkit prody vina prolif numba
pip install -e ".[dev]"
python scripts/build_dunbrack.py  # one-time rotamer library build
```

## Usage

```bash
spade run --uniprot P00533 --ligand "SMILES" --output ./out
spade interactive
```
