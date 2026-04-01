"""
cli.py — Command-line interface for SPADE using typer.

Commands:
  spade run        Full pipeline: fetch → ensemble → dock → report
  spade prep       Fetch and prepare a structure only
  spade dock       Dock against a pre-generated ensemble
  spade interactive  Launch the rich/questionary TUI
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="spade",
    help="SPADE — Structural PAE-Aware Docking Ensembles",
    add_completion=False,
)


@app.command("run")
def cmd_run(
    uniprot: Optional[str] = typer.Option(None, "--uniprot", "-u", help="UniProt ID (e.g. P00533)"),
    pdb: Optional[Path] = typer.Option(None, "--pdb", help="Local PDB file (skip EBI fetch)"),
    pae: Optional[Path] = typer.Option(None, "--pae", help="Local PAE JSON file (required with --pdb)"),
    ligand: str = typer.Option(..., "--ligand", "-l", help="Ligand SMILES string"),
    output: Path = typer.Option(Path("./spade_out"), "--output", "-o", help="Output directory"),
    n_conformers: int = typer.Option(10, "--conformers", "-n", help="Number of ensemble conformers"),
    exhaustiveness: int = typer.Option(8, "--exhaustiveness", "-e", help="Vina exhaustiveness"),
    n_poses: int = typer.Option(9, "--poses", help="Poses per ligand per conformer"),
    ph: float = typer.Option(7.4, "--ph", help="Target pH for protonation state"),
    no_stereo: bool = typer.Option(False, "--no-stereo", help="Disable stereoisomer enumeration"),
) -> None:
    """
    Full SPADE pipeline: structure → flexibility → ensemble → dock → report.
    """
    import numpy as np
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

    console = Console()

    # ------------------------------------------------------------------
    # Load structure
    # ------------------------------------------------------------------
    from spade.core.structure import load_structure, fetch_structure

    if pdb:
        console.print(f"Loading structure from [cyan]{pdb}[/cyan]")
        structure = load_structure(str(pdb), str(pae) if pae else "")
        uid = uniprot or pdb.stem
    elif uniprot:
        console.print(f"Fetching [bold]{uniprot}[/bold] from EBI AlphaFold…")
        structure = fetch_structure(uniprot)
        uid = uniprot
    else:
        typer.echo("Error: provide --uniprot or --pdb", err=True)
        raise typer.Exit(code=1)

    console.print(f"[green]Structure: {structure.n_residues} residues, AF {structure.af_version}[/green]")

    # ------------------------------------------------------------------
    # Flexibility profile + pocket (heuristic: all residues)
    # ------------------------------------------------------------------
    from spade.core.flexibility import build_flexibility_profile

    pocket_residues = np.arange(min(20, structure.n_residues))
    ca_sel = structure.atoms.select("name CA")
    ca_coords = ca_sel.getCoords() if ca_sel is not None else np.zeros((structure.n_residues, 3))
    profile = build_flexibility_profile(
        structure.plddt, structure.pae_matrix, pocket_residues, ca_coords
    )

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------
    from spade.core.ensemble import EnsembleGenerator

    console.print(f"Generating {n_conformers} conformers…")
    gen = EnsembleGenerator(structure, profile, n_conformers=n_conformers)
    conformers = gen.generate()
    console.print(f"[green]{len(conformers)} conformers generated[/green]")

    # ------------------------------------------------------------------
    # Ligand preparation
    # ------------------------------------------------------------------
    from spade.core.ligand import prepare_ligand

    console.print("Preparing ligand…")
    ligands = prepare_ligand(ligand, ph=ph, enumerate_stereo=not no_stereo)
    console.print(f"[green]{len(ligands)} ligand variant(s)[/green]")

    # ------------------------------------------------------------------
    # Docking
    # ------------------------------------------------------------------
    from spade.core.docking import dock_ensemble

    console.print(f"Docking {len(ligands)} variant(s) × {len(conformers)} conformers…")
    docking_results = dock_ensemble(
        conformers, ligands, pocket_residues,
        exhaustiveness=exhaustiveness, n_poses=n_poses,
    )

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    from spade.core.clustering import cluster_poses

    consensus = cluster_poses(docking_results, conformers, ligands[0].mol)
    top = consensus.top_cluster
    console.print(
        f"[bold green]Top score: {top.mean_score:.2f} kcal/mol[/bold green] | "
        f"Clusters: {consensus.n_clusters} | "
        f"Site confidence: {consensus.site_confidence}"
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
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

    prov = RunProvenance(
        uniprot_id=uid,
        af_version=structure.af_version,
        n_residues=structure.n_residues,
        pocket_residues=pocket_residues.tolist(),
        ligand_smiles=ligand,
        n_ligand_variants=len(ligands),
        n_conformers_requested=n_conformers,
        n_conformers_generated=len(conformers),
        exhaustiveness=exhaustiveness,
        n_poses_per_conformer=n_poses,
        n_total_poses=consensus.n_total_poses,
        n_clusters=consensus.n_clusters,
        top_cluster_score=top.mean_score,
        top_cluster_fraction_ensemble=top.fraction_ensemble,
        site_confidence=consensus.site_confidence,
        conformer_summaries=conf_summaries,
        plddt_mean=float(structure.plddt.mean()),
        plddt_std=float(structure.plddt.std()),
        command_line=" ".join(sys.argv),
    )

    generate_report(prov, str(output))
    console.print(f"[green]Report written to [cyan]{output}[/cyan][/green]")


@app.command("prep")
def cmd_prep(
    uniprot: Optional[str] = typer.Option(None, "--uniprot", "-u", help="UniProt ID"),
    output: Path = typer.Option(Path("./spade_prep"), "--output", "-o", help="Output directory"),
) -> None:
    """
    Fetch and save an AlphaFold structure with pLDDT/PAE summary.
    """
    if not uniprot:
        typer.echo("Error: --uniprot required for prep command", err=True)
        raise typer.Exit(code=1)

    from spade.core.structure import fetch_structure, write_structure
    from rich.console import Console

    console = Console()
    with console.status(f"Fetching {uniprot}…"):
        structure = fetch_structure(uniprot)

    output.mkdir(parents=True, exist_ok=True)
    pdb_out = output / f"{uniprot}.pdb"
    write_structure(structure.atoms, str(pdb_out))
    console.print(
        f"[green]Saved: {pdb_out}[/green] | "
        f"{structure.n_residues} residues | "
        f"mean pLDDT {structure.plddt.mean():.1f}"
    )


@app.command("dock")
def cmd_dock(
    ensemble_dir: Path = typer.Option(..., "--ensemble-dir", "-d", help="Directory with PDB conformers"),
    ligand: str = typer.Option(..., "--ligand", "-l", help="Ligand SMILES"),
    output: Path = typer.Option(Path("./spade_dock_out"), "--output", "-o", help="Output directory"),
    exhaustiveness: int = typer.Option(8, "--exhaustiveness"),
    n_poses: int = typer.Option(9, "--poses"),
    ph: float = typer.Option(7.4, "--ph"),
) -> None:
    """
    Dock a ligand against a pre-built ensemble directory (PDB files).
    """
    import numpy as np
    from rich.console import Console
    from spade.core.structure import load_structure
    from spade.core.ligand import prepare_ligand
    from spade.core.docking import dock_ensemble
    from spade.core.clustering import cluster_poses
    from spade.core.report import RunProvenance, ConformerSummary, generate_report

    console = Console()

    pdb_files = sorted(ensemble_dir.glob("*.pdb"))
    if not pdb_files:
        typer.echo(f"No PDB files found in {ensemble_dir}", err=True)
        raise typer.Exit(code=1)

    import prody
    prody.confProDy(verbosity="none")
    conformers = [prody.parsePDB(str(p)) for p in pdb_files]
    console.print(f"Loaded {len(conformers)} conformers from {ensemble_dir}")

    ligands = prepare_ligand(ligand, ph=ph)
    console.print(f"Prepared {len(ligands)} ligand variant(s)")

    # Use all residues as pocket for ad-hoc docking
    n_res = conformers[0].select("name CA").numAtoms() if conformers else 0
    pocket_residues = np.arange(min(20, n_res))

    docking_results = dock_ensemble(
        conformers, ligands, pocket_residues,
        exhaustiveness=exhaustiveness, n_poses=n_poses,
    )
    consensus = cluster_poses(docking_results, conformers, ligands[0].mol)

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

    prov = RunProvenance(
        ligand_smiles=ligand,
        n_ligand_variants=len(ligands),
        n_conformers_generated=len(conformers),
        n_total_poses=consensus.n_total_poses,
        n_clusters=consensus.n_clusters,
        top_cluster_score=consensus.top_cluster.mean_score,
        top_cluster_fraction_ensemble=consensus.top_cluster.fraction_ensemble,
        site_confidence=consensus.site_confidence,
        conformer_summaries=conf_summaries,
        command_line=" ".join(sys.argv),
    )

    generate_report(prov, str(output))
    console.print(
        f"[green]Done. Top score: {consensus.top_cluster.mean_score:.2f} kcal/mol. "
        f"Report: {output}[/green]"
    )


@app.command("interactive")
def cmd_interactive(
    output: Path = typer.Option(Path("./spade_out"), "--output", "-o", help="Output directory"),
) -> None:
    """
    Launch the interactive rich/questionary TUI session.
    """
    from spade.interfaces.tui import run_session
    run_session(output_dir=str(output))


def main() -> None:
    app()
