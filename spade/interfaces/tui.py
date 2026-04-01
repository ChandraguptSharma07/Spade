"""
tui.py — Interactive terminal UI for SPADE using rich + questionary.

Flow:
  1. UniProt ID prompt → fetch/load structure → pLDDT summary + PAE warnings
  2. Pocket selection (auto-detect / residue list / reference ligand file)
  3. Ligand SMILES prompt → stereocenter/tautomer info
  4. Ensemble size selection (5 / 10 / 20)
  5. Run with per-conformer rich progress bar
  6. Final result summary in rich Panel
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

try:
    import questionary
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich import print as rprint
except ImportError as e:
    raise ImportError(
        "rich and questionary are required: pip install rich questionary"
    ) from e

import numpy as np

console = Console()


def run_session(
    output_dir: str = "./spade_out",
    pdb_path: Optional[str] = None,
    pae_path: Optional[str] = None,
) -> None:
    """
    Entry point for the interactive TUI session.

    Parameters
    ----------
    output_dir : where to write provenance.json and report.html
    pdb_path   : if provided, skip the UniProt fetch and load from disk
    pae_path   : if provided with pdb_path, load PAE from this JSON file
    """
    console.rule("[bold blue]SPADE[/bold blue] — Structural PAE-Aware Docking Ensembles")
    console.print()

    # -------------------------------------------------------------------------
    # Step 1 — Structure
    # -------------------------------------------------------------------------
    structure, uniprot_id = _step_structure(pdb_path, pae_path)
    if structure is None:
        console.print("[red]Could not load structure. Aborting.[/red]")
        sys.exit(1)

    _show_plddt_summary(structure)

    # -------------------------------------------------------------------------
    # Step 2 — Pocket
    # -------------------------------------------------------------------------
    pocket_residues = _step_pocket(structure)

    # -------------------------------------------------------------------------
    # Step 3 — Ligand
    # -------------------------------------------------------------------------
    smiles, ligands = _step_ligand()
    if not ligands:
        console.print("[red]Ligand preparation failed. Aborting.[/red]")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 4 — Ensemble size
    # -------------------------------------------------------------------------
    n_conformers = _step_ensemble_size()

    # -------------------------------------------------------------------------
    # Step 5 — Run
    # -------------------------------------------------------------------------
    results = _step_run(structure, pocket_residues, ligands, n_conformers)
    if results is None:
        return

    conformers, docking_results, consensus = results

    # -------------------------------------------------------------------------
    # Step 6 — Report
    # -------------------------------------------------------------------------
    _step_report(
        structure=structure,
        uniprot_id=uniprot_id,
        smiles=smiles,
        ligands=ligands,
        conformers=conformers,
        docking_results=docking_results,
        pocket_residues=pocket_residues,
        consensus=consensus,
        n_conformers=n_conformers,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def _step_structure(pdb_path, pae_path):
    """Load or fetch structure. Returns (AlphaFoldStructure, uniprot_id)."""
    from spade.core.structure import AlphaFoldStructure, load_structure, fetch_structure

    if pdb_path:
        uniprot_id = Path(pdb_path).stem
        console.print(f"Loading structure from [cyan]{pdb_path}[/cyan]")
        try:
            structure = load_structure(pdb_path, pae_path or "")
            return structure, uniprot_id
        except Exception as exc:
            console.print(f"[red]Error loading structure: {exc}[/red]")
            return None, uniprot_id

    uniprot_id = questionary.text(
        "UniProt ID (e.g. P00533 for EGFR):",
        validate=lambda v: len(v.strip()) >= 4 or "Enter a valid UniProt ID",
    ).ask()

    if not uniprot_id:
        return None, ""
    uniprot_id = uniprot_id.strip().upper()

    with console.status(f"Fetching [bold]{uniprot_id}[/bold] from EBI AlphaFold DB…"):
        try:
            structure = fetch_structure(uniprot_id)
            console.print(f"[green]Loaded {structure.n_residues} residues[/green]")
            return structure, uniprot_id
        except Exception as exc:
            console.print(f"[red]Fetch failed: {exc}[/red]")
            return None, uniprot_id


def _show_plddt_summary(structure) -> None:
    """Print pLDDT tier breakdown and PAE warnings."""
    from spade.core.flexibility import classify_residues

    plddt = structure.plddt
    tiers = classify_residues(plddt)

    counts = {"rigid": 0, "moderate": 0, "flexible": 0, "disordered": 0}
    for t in tiers.values():
        counts[t] = counts.get(t, 0) + 1

    table = Table(title="pLDDT Summary", show_header=True, header_style="bold")
    table.add_column("Tier")
    table.add_column("Count", justify="right")
    table.add_column("Fraction", justify="right")
    n = len(plddt)
    for tier_name, colour in [("rigid", "green"), ("moderate", "yellow"), ("flexible", "orange3"), ("disordered", "red")]:
        c = counts.get(tier_name, 0)
        table.add_row(f"[{colour}]{tier_name}[/{colour}]", str(c), f"{c/n*100:.1f}%")
    console.print(table)

    # PAE warning — check for off-diagonal low-PAE blocks
    pae = structure.pae_matrix
    n_res = pae.shape[0]
    if n_res > 50:
        # Simple heuristic: check if any far-apart residue pair has PAE < 5
        far_pairs = pae[np.triu_indices(n_res, k=30)]
        if far_pairs.min() < 5.0:
            console.print(
                "[yellow]Warning: low PAE values between distant residues detected — "
                "possible multi-domain protein. Flexibility graph restricted to pocket-local region.[/yellow]"
            )


def _step_pocket(structure) -> np.ndarray:
    """Pocket selection: auto / manual residue list."""
    method = questionary.select(
        "Pocket selection method:",
        choices=[
            "Auto-detect (highest pLDDT cluster)",
            "Specify residue indices (0-based, comma-separated)",
        ],
    ).ask()

    if method is None:
        return np.arange(min(20, structure.n_residues))

    if "Auto" in method:
        pocket = _auto_pocket(structure)
        console.print(f"Auto-detected pocket: {len(pocket)} residues")
        return pocket

    raw = questionary.text("Residue indices (e.g. 1,2,5,10):").ask() or ""
    try:
        indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
        if not indices:
            raise ValueError
        return np.array(indices)
    except ValueError:
        console.print("[yellow]Could not parse indices — using auto-detect.[/yellow]")
        return _auto_pocket(structure)


def _auto_pocket(structure) -> np.ndarray:
    """Heuristic: take the 20 residues around the pLDDT-highest region."""
    plddt = structure.plddt
    center = int(np.argmax(plddt))
    half = 10
    start = max(0, center - half)
    end = min(len(plddt), center + half)
    return np.arange(start, end)


def _step_ligand():
    """Prompt for SMILES and prepare ligand variants."""
    from spade.core.ligand import prepare_ligand

    smiles = questionary.text(
        "Ligand SMILES:",
        validate=lambda v: len(v.strip()) > 2 or "Enter a valid SMILES string",
    ).ask()

    if not smiles:
        return "", []

    smiles = smiles.strip()

    with console.status("Preparing ligand (tautomers, stereoisomers, 3D, PDBQT)…"):
        try:
            ligands = prepare_ligand(smiles)
        except Exception as exc:
            console.print(f"[red]Ligand preparation failed: {exc}[/red]")
            return smiles, []

    console.print(
        f"[green]Prepared {len(ligands)} ligand variant(s)[/green] "
        f"(tautomers × stereoisomers × protomers)"
    )

    # Stereocenter info
    undefined = [l for l in ligands if l.n_undefined_stereocenters > 0]
    if undefined:
        console.print(
            f"[yellow]{len(undefined)} variant(s) had undefined stereocenters — "
            f"all enantiomers enumerated.[/yellow]"
        )

    return smiles, ligands


def _step_ensemble_size() -> int:
    raw = questionary.select(
        "Ensemble size (number of receptor conformers):",
        choices=["5", "10", "20"],
        default="10",
    ).ask()
    return int(raw or "10")


def _step_run(structure, pocket_residues, ligands, n_conformers):
    """Generate ensemble and dock."""
    from spade.core.flexibility import build_flexibility_profile
    from spade.core.ensemble import EnsembleGenerator
    from spade.core.docking import dock_ensemble
    from spade.core.clustering import cluster_poses

    # Flexibility profile
    with console.status("Computing PAE-weighted flexibility profile…"):
        ca_sel = structure.atoms.select("name CA")
        ca_coords = ca_sel.getCoords() if ca_sel is not None else np.zeros((structure.n_residues, 3))
        profile = build_flexibility_profile(
            structure.plddt, structure.pae_matrix, pocket_residues, ca_coords
        )

    # Ensemble generation
    console.print(f"\nGenerating {n_conformers} conformers via PAE-weighted NMA…")
    gen = EnsembleGenerator(structure, profile, n_conformers=n_conformers)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating conformers…", total=n_conformers)
        conformers = gen.generate()
        progress.update(task, completed=len(conformers))

    console.print(f"[green]Generated {len(conformers)} conformers[/green]")

    # Docking
    if not _check_vina():
        console.print(
            "[yellow]AutoDock Vina not available — skipping docking.[/yellow]\n"
            "Install with: conda install -c conda-forge vina"
        )
        return None

    console.print(f"\nDocking {len(ligands)} ligand variant(s) × {len(conformers)} conformers…")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        total = len(conformers) * len(ligands)
        task = progress.add_task("Docking…", total=total)

        docking_results = []
        for ci, conformer in enumerate(conformers):
            from spade.core.docking import dock_ensemble as _de, compute_bounding_box
            bbox = compute_bounding_box(conformer, pocket_residues)
            from spade.core.docking import _dock_single, DockingResult
            import time as _time
            for ligand in ligands:
                t0 = _time.perf_counter()
                from spade.core.docking import _dock_single
                from vina import Vina
                try:
                    poses = _dock_single(conformer, ligand, bbox, 8, 9, ci)
                except Exception:
                    poses = []
                docking_results.append(DockingResult(
                    conformer_index=ci,
                    conformer_ca_rmsd=float(conformer.getData("ca_rmsd_from_ref")[0]),
                    poses=poses,
                    bounding_box=bbox,
                    docking_time_seconds=_time.perf_counter() - t0,
                ))
                progress.advance(task)

    # Clustering
    with console.status("Clustering poses by PLIF fingerprints…"):
        ligand_mol = ligands[0].mol if ligands else None
        consensus = cluster_poses(docking_results, conformers, ligand_mol)

    return conformers, docking_results, consensus


def _check_vina() -> bool:
    try:
        from vina import Vina  # noqa: F401
        return True
    except ImportError:
        return False


def _step_report(
    structure, uniprot_id, smiles, ligands, conformers, docking_results,
    pocket_residues, consensus, n_conformers, output_dir
) -> None:
    """Write reports and show final summary panel."""
    from spade.core.report import RunProvenance, ConformerSummary, generate_report

    conf_summaries = []
    for dr in docking_results:
        scores = [p.score_kcal_mol for p in dr.poses] if dr.poses else [0.0]
        conf_summaries.append(ConformerSummary(
            conformer_index=dr.conformer_index,
            ca_rmsd_from_ref=dr.conformer_ca_rmsd,
            n_poses=len(dr.poses),
            best_score_kcal_mol=min(scores),
            docking_time_seconds=dr.docking_time_seconds,
        ))

    plddt = structure.plddt
    prov = RunProvenance(
        uniprot_id=uniprot_id,
        af_version=structure.af_version,
        n_residues=structure.n_residues,
        pocket_residues=pocket_residues.tolist(),
        ligand_smiles=smiles,
        n_ligand_variants=len(ligands),
        n_conformers_requested=n_conformers,
        n_conformers_generated=len(conformers),
        n_total_poses=consensus.n_total_poses,
        n_clusters=consensus.n_clusters,
        top_cluster_score=consensus.top_cluster.mean_score,
        top_cluster_fraction_ensemble=consensus.top_cluster.fraction_ensemble,
        site_confidence=consensus.site_confidence,
        conformer_summaries=conf_summaries,
        plddt_mean=float(plddt.mean()),
        plddt_std=float(plddt.std()),
        inter_domain_pae_warning=getattr(
            _build_profile_cached(structure, pocket_residues), "inter_domain_pae_warning", False
        ),
    )

    with console.status(f"Writing report to [cyan]{output_dir}[/cyan]…"):
        generate_report(prov, output_dir)

    top = consensus.top_cluster
    panel_content = (
        f"[bold green]Top cluster score:[/bold green] {top.mean_score:.2f} kcal/mol\n"
        f"[bold]Ensemble coverage:[/bold] {top.fraction_ensemble*100:.0f}% of conformers\n"
        f"[bold]Site confidence:[/bold] {consensus.site_confidence.upper()}\n"
        f"[bold]Total poses clustered:[/bold] {consensus.n_total_poses}\n"
        f"[bold]Clusters found:[/bold] {consensus.n_clusters}\n\n"
        f"Report written to: [cyan]{output_dir}/[/cyan]"
    )
    console.print()
    console.print(Panel(panel_content, title="SPADE Results", border_style="green"))


def _build_profile_cached(structure, pocket_residues):
    """Helper to fetch a FlexibilityProfile without re-running the whole pipeline."""
    try:
        from spade.core.flexibility import build_flexibility_profile
        ca_sel = structure.atoms.select("name CA")
        ca_coords = ca_sel.getCoords() if ca_sel is not None else np.zeros((structure.n_residues, 3))
        return build_flexibility_profile(
            structure.plddt, structure.pae_matrix, pocket_residues, ca_coords
        )
    except Exception:
        return type("_FP", (), {"inter_domain_pae_warning": False})()
