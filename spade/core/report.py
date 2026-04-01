"""
report.py — Pipeline run provenance and HTML report generation.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except ImportError as e:
    raise ImportError("Jinja2 is required: pip install jinja2") from e

from spade.core.clustering import ConsensusResult

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


@dataclass
class ConformerSummary:
    """Per-conformer docking summary for the report table."""
    conformer_index: int
    ca_rmsd_from_ref: float
    n_poses: int
    best_score_kcal_mol: float
    docking_time_seconds: float


@dataclass
class RunProvenance:
    """Complete metadata for one SPADE pipeline run."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    spade_version: str = "0.1.0"
    uniprot_id: str = ""
    af_version: str = ""
    n_residues: int = 0
    pocket_residues: list[int] = field(default_factory=list)
    ligand_smiles: str = ""
    n_ligand_variants: int = 0
    n_conformers_requested: int = 0
    n_conformers_generated: int = 0
    exhaustiveness: int = 8
    n_poses_per_conformer: int = 9
    n_total_poses: int = 0
    n_clusters: int = 0
    top_cluster_score: float = 0.0
    top_cluster_fraction_ensemble: float = 0.0
    site_confidence: str = "unknown"
    conformer_summaries: list[ConformerSummary] = field(default_factory=list)
    plddt_mean: float = 0.0
    plddt_std: float = 0.0
    inter_domain_pae_warning: bool = False
    command_line: str = ""


def generate_report(provenance: RunProvenance, output_dir: str) -> None:
    """
    Write provenance.json and report.html to output_dir.

    Parameters
    ----------
    provenance  : RunProvenance with all pipeline metadata
    output_dir  : Path to write outputs (created if absent)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Write provenance JSON
    prov_path = out / "provenance.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(_provenance_to_dict(provenance), f, indent=2)

    # Render HTML report
    html = _render_html(provenance)
    html_path = out / "report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def _provenance_to_dict(prov: RunProvenance) -> dict:
    """Convert RunProvenance to a JSON-serialisable dict."""
    d = asdict(prov)
    return d


def _render_html(prov: RunProvenance) -> str:
    """Render the Jinja2 HTML report template."""
    if not _TEMPLATES_DIR.exists():
        return _minimal_html(prov)

    try:
        env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=select_autoescape(["html"]),
        )
        tmpl = env.get_template("report.html.j2")
        return tmpl.render(prov=prov, prov_json=json.dumps(_provenance_to_dict(prov), indent=2))
    except Exception:
        return _minimal_html(prov)


def _minimal_html(prov: RunProvenance) -> str:
    """Minimal HTML fallback if Jinja2 template is missing."""
    prov_json = json.dumps(_provenance_to_dict(prov), indent=2)
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>SPADE Report — {prov.uniprot_id}</title></head>
<body>
<h1>SPADE Docking Report</h1>
<p><strong>UniProt:</strong> {prov.uniprot_id} &nbsp;
   <strong>Run ID:</strong> {prov.run_id} &nbsp;
   <strong>Timestamp:</strong> {prov.timestamp}</p>
<h2>Results</h2>
<ul>
  <li>Ligand: {prov.ligand_smiles}</li>
  <li>Conformers generated: {prov.n_conformers_generated}</li>
  <li>Total poses: {prov.n_total_poses}</li>
  <li>Clusters: {prov.n_clusters}</li>
  <li>Top cluster score: {prov.top_cluster_score:.2f} kcal/mol</li>
  <li>Site confidence: {prov.site_confidence}</li>
</ul>
<h2>Provenance</h2>
<script type="application/json" id="spade-provenance">
{prov_json}
</script>
<pre>{prov_json}</pre>
</body>
</html>"""
