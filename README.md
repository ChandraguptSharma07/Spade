# SPADE: Structural PAE-Aware Docking Ensembles

[![PyPI](https://img.shields.io/pypi/v/spade-docking)](https://pypi.org/project/spade-docking/)
[![CI](https://github.com/ChandraguptSharma07/Spade/actions/workflows/ci.yml/badge.svg)](https://github.com/ChandraguptSharma07/Spade/actions/workflows/ci.yml)
[![Docker](https://ghcr.io/ChandraguptSharma07/spade:latest)](https://github.com/ChandraguptSharma07/Spade/pkgs/container/spade)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChandraguptSharma07/Spade/blob/main/notebooks/SPADE_demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Ensemble docking pipeline that uses AlphaFold2's confidence outputs instead of ignoring them.

Fetch AF2 structure → classify residues by pLDDT → build PAE-weighted flexibility graph (pocket-local) → NMA conformer ensemble → dock each conformer with Vina → cluster poses by PLIF interaction fingerprint → consensus score + uncertainty estimate.

## Install

**Option 1 — pip (recommended)**

```bash
pip install spade-docking
```

Requires Python 3.10+. Some compiled dependencies (RDKit, ProDy, AutoDock Vina) are easiest to get via conda first:

```bash
conda install -c conda-forge rdkit prody vina prolif numba
pip install spade-docking
```

**Option 2 — conda (full environment)**

```bash
conda env create -f environment.yml
conda activate spade
```

**Option 3 — Docker**

```bash
docker run --rm \
  -v $(pwd)/out:/workspace \
  ghcr.io/chandraguptsharma07/spade:latest \
  spade run --uniprot P00533 --ligand "CCc1cc2sc(NC(=O)Nc3ccc(F)cc3)nc2cc1" --output /workspace
```

**Google Colab** — open the demo notebook with the badge above; the first cell handles all installation automatically.

## Quick Start

```bash
# One-liner: fetch EGFR, dock erlotinib, write report to ./out/
spade run --uniprot P00533 --ligand "CCc1cc2sc(NC(=O)Nc3ccc(F)cc3)nc2cc1" --output ./out

# Guided interactive session
spade interactive

# Stage-by-stage
spade prep --uniprot P00533 --output ./out
spade dock --ensemble-dir ./out --ligand "CCc1..." --output ./out
```

## Output

```
out/
├── provenance.json   # full run metadata (pipeline version, params, timings)
├── report.html       # interactive HTML report with score table + 3D viewer
└── poses/            # PDBQT poses for every conformer × pose
```

## How it works

| Step | Module | Key idea |
|------|--------|----------|
| Structure fetch | `structure.py` | AF2 PDB + PAE matrix from EBI |
| Flexibility profiling | `flexibility.py` | pLDDT tiers + pocket-local PAE graph (12 Å cutoff) |
| Ensemble generation | `ensemble.py` | ANM normal modes weighted by PAE; Cα RMSD cap 1.2 Å |
| Ligand preparation | `ligand.py` | RDKit tautomers + Dimorphite protomers + stereo enumeration |
| Docking | `docking.py` | Per-conformer bounding box + AutoDock Vina |
| Clustering | `clustering.py` | PLIF Tanimoto + DBSCAN — **not** RMSD across conformers |
| Report | `report.py` | Provenance JSON + Jinja2 HTML |

## Citation

If you use SPADE in published work, please cite this repository until a preprint is available.

## License

MIT
