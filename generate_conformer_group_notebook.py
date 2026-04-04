"""
generate_conformer_group_notebook.py

Generates notebooks/ABL1_ConditionD_GroupAnalysis.ipynb

Purpose: Run ONLY Condition D (domain rearrangement + Phe382 chi1) and
break down clustering by conformer group (DFG-in / DFG-mid / DFG-out).

No NMA conditions. Saves ~70% of the runtime vs the full 4-way notebook.
"""

import json, pathlib

NB = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}, "cells": []}

def code(src, cell_id):
    NB["cells"].append({
        "cell_type": "code", "id": cell_id,
        "metadata": {}, "outputs": [], "execution_count": None,
        "source": src if isinstance(src, str) else "\n".join(src),
    })

def md(src, cell_id):
    NB["cells"].append({
        "cell_type": "markdown", "id": cell_id,
        "metadata": {},
        "source": src if isinstance(src, str) else "\n".join(src),
    })

# ── Cell 0: Install ───────────────────────────────────────────────────────────
md("# ABL1 — Condition D: Per-Conformer-Group Clustering\n\n"
   "Runs only the domain rearrangement condition (backbone rotation + Phe382 chi1 = −90°) "
   "and breaks down clustering by DFG-in / DFG-intermediate / DFG-out conformer groups.\n\n"
   "**Question:** Does Imatinib cluster consistently in DFG-out conformers specifically, "
   "while fragmenting in DFG-in conformers?", "cell-md-title")

code("""\
import subprocess, sys

# Pull latest SPADE from GitHub (includes chi1 fix)
subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
    "git+https://github.com/ChandraguptSharma07/Spade.git"
], check=True)

# UniDock binary
import os, urllib.request, stat, pathlib
UNIDOCK_URL = "https://github.com/dptech-corp/Uni-Dock/releases/download/1.1.2/unidock"
UNIDOCK_BIN = pathlib.Path("/usr/local/bin/unidock")
if not UNIDOCK_BIN.exists():
    print("Downloading UniDock binary...")
    urllib.request.urlretrieve(UNIDOCK_URL, UNIDOCK_BIN)
    UNIDOCK_BIN.chmod(UNIDOCK_BIN.stat().st_mode | stat.S_IEXEC)
    print("UniDock ready.")
else:
    print("UniDock already present.")
print("Setup complete.")
""", "cell-install")

# ── Cell 1: Imports & constants ───────────────────────────────────────────────
code("""\
import warnings, pathlib
import numpy as np
warnings.filterwarnings("ignore")

from spade.core.structure    import AlphaFoldStructure
from spade.core.ligand       import SmilesMolecule
from spade.core.docking      import dock_ensemble
from spade.core.clustering   import cluster_poses
from spade.core.domain_rearrangement import (
    PAEDomainRearrangementGenerator,
    identify_mobile_segment,
)

# ── ABL1 (UniProt P00519) ─────────────────────────────────────────────────────
UNIPROT_ID  = "P00519"

# Pocket residues: ATP-binding site + DFG-adjacent (0-based indices)
# Includes Thr315 (gatekeeper), DFG loop, P-loop, hinge
POCKET_RESIDUES = list(range(248, 260)) + list(range(315, 320)) + list(range(380, 405))

# N-lobe reference domain for inter-domain PAE (stable β-sheet core, res 235–310)
NLOBE_RESNUMS   = list(range(235, 311))

# Kinase domain search range (avoids unrelated high-PAE regions)
KINASE_DOMAIN   = list(range(230, 510))

# PAE calibration from ABL1 2GQG/2HYY crystal pair
PAE_SCALE_FACTOR   = 1.85
N_CONFORMERS       = 10
ROTATION_DIRECTION = 1   # CCW = DFG-out direction for ABL1
SEEDS_BASE         = 42  # seed for conformer i = SEEDS_BASE + i

# Ligands
LIGANDS = {
    "Imatinib":  "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",
    "Dasatinib": "Cc1nc(Nc2ncc(s2)C(=O)Nc3c(C)cccc3Cl)cc(n1)N4CCN(CCO)CC4",
}

# Conformer groups for per-group analysis
CONFORMER_GROUPS = {
    "DFG-in  (conf 0-2,  θ ≈  0-11°)":  [0, 1, 2],
    "DFG-mid (conf 3-6,  θ ≈ 17-33°)":  [3, 4, 5, 6],
    "DFG-out (conf 7-9,  θ ≈ 38-50°)":  [7, 8, 9],
}

OUT_DIR = pathlib.Path("abl1_4way_experiment")
OUT_DIR.mkdir(exist_ok=True)

print("Constants loaded.")
print(f"  Ligands     : {list(LIGANDS.keys())}")
print(f"  N conformers: {N_CONFORMERS}")
print(f"  Scale factor: {PAE_SCALE_FACTOR}")
""", "cell-constants")

# ── Cell 2: Fetch structure ───────────────────────────────────────────────────
code("""\
print(f"Fetching AlphaFold structure for {UNIPROT_ID}...")
abl1 = AlphaFoldStructure(UNIPROT_ID)
print(f"  Residues : {abl1.n_residues}")
print(f"  PAE matrix: {abl1.pae_matrix.shape}")
""", "cell-structure")

# ── Cell 3: Prepare ligands ───────────────────────────────────────────────────
code("""\
print("Preparing ligands...")
prepared = {}
for name, smiles in LIGANDS.items():
    mol = SmilesMolecule(smiles, name=name)
    prepared[name] = mol.prepare()
    print(f"  {name}: {prepared[name].n_heavy_atoms} heavy atoms")
""", "cell-ligands")

# ── Cell 4: Identify segment + generate conformers ────────────────────────────
code("""\
print("\\nIdentifying mobile segment from inter-domain PAE...")
segment = identify_mobile_segment(
    structure              = abl1,
    reference_domain_resnums = NLOBE_RESNUMS,
    min_inter_pae          = 6.0,
    min_segment_length     = 5,
    max_segment_length     = 60,
    min_plddt              = 50.0,
    search_resnums         = KINASE_DOMAIN,
)
assert segment is not None, "No mobile segment found — check NLOBE_RESNUMS / KINASE_DOMAIN"

ca_sel   = abl1.atoms.select("calpha")
resnums  = ca_sel.getResnums()
seg_res  = [int(resnums[i]) for i in segment.residues]
print(f"  Mobile segment : res {seg_res[0]}–{seg_res[-1]} ({len(seg_res)} residues)")
print(f"  Hinge          : Cα({segment.hinge_n_resnum}) → Cα({segment.hinge_c_resnum})")
print(f"  Mean inter-PAE : {segment.mean_inter_pae:.2f} Å")

print("\\nGenerating Condition D conformers (backbone rotation + Phe382 chi1 = -90°)...")
gen_D = PAEDomainRearrangementGenerator(
    structure          = abl1,
    mobile_segment     = segment,
    inter_domain_pae   = segment.mean_inter_pae,
    pae_scale_factor   = PAE_SCALE_FACTOR,
    n_conformers       = N_CONFORMERS,
    rotation_direction = ROTATION_DIRECTION,
    sidechain_targets  = {382: -90.0},   # Phe382 chi1 → DFG-out rotamer
)

conformers_D = gen_D.generate()
print(f"  theta_max = {gen_D.theta_max_deg:.1f}°")
print(f"  {len(conformers_D)} conformers generated")

# Measure Phe382 Cα displacement per conformer
ref_coords   = abl1.atoms.select("calpha and resnum 382").getCoords()[0]
print("\\n  Phe382 Cα displacement (target: 6.81 Å):")
for i, conf in enumerate(conformers_D):
    ca = conf.select("calpha and resnum 382")
    if ca is not None:
        disp = float(np.linalg.norm(ca.getCoords()[0] - ref_coords))
        angle = i * gen_D.theta_max_deg / (N_CONFORMERS - 1)
        bar = "#" * int(disp * 3)
        print(f"    Conf {i:2d} (θ={angle:5.1f}°): {disp:.2f} Å  {bar}")
""", "cell-conformers")

# ── Cell 5: Dock ──────────────────────────────────────────────────────────────
code("""\
import time

print("\\nDocking Condition D against all conformers...")
print("(sequential execution — no GPU deadlock)")

results_D = {}   # ligand_name → list[DockingResult]

for lig_name, lig_prep in prepared.items():
    print(f"  Docking {lig_name}...")
    t0 = time.time()
    dr_list = dock_ensemble(
        conformers    = conformers_D,
        ligands       = [lig_prep],
        pocket_residues = np.array(POCKET_RESIDUES),
        backend       = "gpu",
        seeds         = [SEEDS_BASE + i for i in range(N_CONFORMERS)],
        n_poses       = 9,
    )
    results_D[lig_name] = [dr for dr in dr_list if dr.poses]
    elapsed = time.time() - t0
    best = min(p.score_kcal_mol for dr in results_D[lig_name] for p in dr.poses)
    print(f"    Best score: {best:.3f} kcal/mol ({elapsed:.0f}s)")

print("\\nDocking complete.")
""", "cell-docking")

# ── Cell 6: Global clustering (baseline for comparison) ──────────────────────
code("""\
print("\\n── GLOBAL CLUSTERING (all conformers pooled) ──")
print(f"  {'Ligand':<12} {'N Clusters':>10} {'Best Score':>12}")
print(f"  {'-'*12} {'-'*10} {'-'*12}")

global_summary = {}
for lig_name, dr_list in results_D.items():
    cr = cluster_poses(dr_list)
    best = min(p.score_kcal_mol for dr in dr_list for p in dr.poses)
    print(f"  {lig_name:<12} {len(cr.clusters):>10} {best:>12.3f}")
    global_summary[lig_name] = {"global_clusters": len(cr.clusters), "global_best": best}
""", "cell-global-cluster")

# ── Cell 7: Per-conformer-group clustering (main analysis) ───────────────────
code("""\
print("\\n── PER-CONFORMER-GROUP CLUSTERING ──")
print()

group_summary = {}   # ligand → group_label → {clusters, best}

for lig_name, dr_list in results_D.items():
    print(f"  {lig_name}")
    print(f"  {'Group':<38} {'Clusters':>8} {'Best Score':>12} {'Interpretation'}")
    print(f"  {'-'*38} {'-'*8} {'-'*12} {'-'*20}")

    group_summary[lig_name] = {}

    for group_label, conf_indices in CONFORMER_GROUPS.items():
        subset = [dr for dr in dr_list if dr.conformer_index in conf_indices]
        if not subset:
            print(f"  {group_label:<38} {'—':>8} {'—':>12}")
            continue

        cr   = cluster_poses(subset)
        n_cl = len(cr.clusters)
        best = min(p.score_kcal_mol for dr in subset for p in dr.poses)

        # Interpretation hint
        if lig_name == "Imatinib":
            hint = "✓ home pocket" if "DFG-out" in group_label and n_cl <= 5 else (
                   "✗ no pocket"   if "DFG-in"  in group_label else "")
        else:  # Dasatinib
            hint = "✓ home state"  if "DFG-in"  in group_label and n_cl <= 5 else (
                   "✗ displaced"   if "DFG-out" in group_label else "")

        print(f"  {group_label:<38} {n_cl:>8} {best:>12.3f}  {hint}")
        group_summary[lig_name][group_label] = {"clusters": n_cl, "best": best}

    print()

# Key comparison: DFG-out clusters for Imatinib vs Dasatinib
print("── KEY COMPARISON ──")
for lig_name in results_D:
    for g_label, vals in group_summary[lig_name].items():
        if "DFG-out" in g_label:
            print(f"  {lig_name:<12} DFG-out clusters: {vals['clusters']:>3}  "
                  f"best: {vals['best']:.3f} kcal/mol")
""", "cell-group-cluster")

# ── Cell 8: Per-conformer best score ─────────────────────────────────────────
code("""\
print("\\n── BEST SCORE PER CONFORMER ──")

for lig_name, dr_list in results_D.items():
    print(f"\\n  {lig_name}")
    print(f"  {'Conf':>5} {'θ (deg)':>8} {'Best Score':>12} {'Bar'}")
    print(f"  {'-'*5} {'-'*8} {'-'*12}")

    for dr in sorted(dr_list, key=lambda x: x.conformer_index):
        if not dr.poses:
            continue
        best = min(dr.poses, key=lambda p: p.score_kcal_mol)
        angle = dr.conformer_index * gen_D.theta_max_deg / (N_CONFORMERS - 1)
        bar = "█" * max(0, int((-best.score_kcal_mol - 7) * 4))
        print(f"  {dr.conformer_index:>5} {angle:>8.1f}° {best.score_kcal_mol:>12.3f}  {bar}")
""", "cell-per-conformer")

# ── Cell 9: Save results ──────────────────────────────────────────────────────
code("""\
import csv, datetime

out_path = OUT_DIR / "conditionD_group_analysis.csv"
rows = []

for lig_name in results_D:
    # Global row
    gs = global_summary[lig_name]
    rows.append({
        "Ligand":       lig_name,
        "Group":        "ALL (global)",
        "N_Clusters":   gs["global_clusters"],
        "Best_Score":   round(gs["global_best"], 3),
    })
    # Per-group rows
    for g_label, vals in group_summary.get(lig_name, {}).items():
        rows.append({
            "Ligand":     lig_name,
            "Group":      g_label.strip(),
            "N_Clusters": vals["clusters"],
            "Best_Score": round(vals["best"], 3),
        })

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Ligand", "Group", "N_Clusters", "Best_Score"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Results saved to {out_path}")
print()
print("SUMMARY TABLE:")
print(f"  {'Ligand':<12} {'Group':<42} {'Clusters':>8} {'Best':>8}")
print(f"  {'-'*12} {'-'*42} {'-'*8} {'-'*8}")
for row in rows:
    print(f"  {row['Ligand']:<12} {row['Group']:<42} {row['N_Clusters']:>8} {row['Best_Score']:>8}")
""", "cell-save")

# ── Write notebook ────────────────────────────────────────────────────────────
out = pathlib.Path("notebooks/ABL1_ConditionD_GroupAnalysis.ipynb")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(NB, indent=1))
print(f"Notebook written to {out}")
