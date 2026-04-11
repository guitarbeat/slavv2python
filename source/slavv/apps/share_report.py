"""Helpers for SLAVV share-report generation and local event logging."""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from plotly.io import to_html

from slavv.analysis import calculate_network_statistics
from slavv.visualization import NetworkVisualizer

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

DEFAULT_SHARE_REPORT_EVENT_LOG = Path(tempfile.gettempdir()) / "slavv_share_report_events.jsonl"


def _format_share_report_value(value: Any) -> str:
    """Format values for compact HTML display."""
    if isinstance(value, bool):
        return "Enabled" if value else "Disabled"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_share_report_value(item) for item in value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _build_share_report_parameter_rows(
    parameters: Mapping[str, Any],
) -> list[tuple[str, str]]:
    """Return a compact subset of processing parameters for the report."""
    selected_keys = [
        ("Voxel size (um)", parameters.get("microns_per_voxel", [1.0, 1.0, 1.0])),
        (
            "Smallest vessel radius (um)",
            parameters.get("radius_of_smallest_vessel_in_microns", "n/a"),
        ),
        (
            "Largest vessel radius (um)",
            parameters.get("radius_of_largest_vessel_in_microns", "n/a"),
        ),
        ("Scales per octave", parameters.get("scales_per_octave", "n/a")),
        ("PSF correction", parameters.get("approximating_PSF", False)),
        ("Edges per vertex", parameters.get("number_of_edges_per_vertex", "n/a")),
    ]
    return [(label, _format_share_report_value(value)) for label, value in selected_keys]


def compute_shareable_stats(
    processing_results: Mapping[str, Any], image_shape: tuple[int, int, int] | None = None
) -> dict[str, Any]:
    """Compute headline network statistics for the share report."""
    parameters = processing_results.get("parameters", {})
    vertices = processing_results["vertices"]
    edges = processing_results["edges"]
    network = processing_results["network"]

    return cast(
        "dict[str, Any]",
        calculate_network_statistics(
            network["strands"],
            network["bifurcations"],
            vertices["positions"],
            vertices.get("radii_microns", vertices.get("radii", [])),
            parameters.get("microns_per_voxel", [1.0, 1.0, 1.0]),
            image_shape or (100, 100, 50),
            edge_energies=edges.get("energies"),
        ),
    )


def build_share_report_signature(
    dataset_name: str, processing_results: Mapping[str, Any], stats: Mapping[str, Any]
) -> str:
    """Build a stable signature used to dedupe report-ready tracking in a session."""
    payload = {
        "dataset_name": dataset_name,
        "vertices": len(processing_results["vertices"].get("positions", [])),
        "edges": len(processing_results["edges"].get("traces", [])),
        "strands": len(processing_results["network"].get("strands", [])),
        "bifurcations": len(processing_results["network"].get("bifurcations", [])),
        "total_length": round(float(stats.get("total_length", 0.0)), 3),
        "volume_fraction": round(float(stats.get("volume_fraction", 0.0)), 6),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def make_share_report_filename(dataset_name: str) -> str:
    """Return a filesystem-friendly report name."""
    stem = Path(dataset_name or "slavv_run").stem
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem).strip("_")
    return f"{safe or 'slavv_run'}_share_report.html"


def _build_share_report_distribution_figure(
    visualizer: NetworkVisualizer, results: Mapping[str, Any]
):
    """Build a secondary figure for the report, with a resilient fallback."""
    try:
        return visualizer.plot_length_weighted_histograms(
            results["vertices"],
            results["edges"],
            results["parameters"],
            number_of_bins=40,
        )
    except Exception:
        return visualizer.plot_radius_distribution(results["vertices"])


def build_share_report_html(
    processing_results: Mapping[str, Any],
    dataset_name: str,
    image_shape: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Build a self-contained HTML share report and related metadata."""
    results = dict(processing_results)
    parameters = results.get("parameters", {})
    stats = compute_shareable_stats(results, image_shape=image_shape)
    visualizer = NetworkVisualizer()

    network_fig = visualizer.plot_3d_network(
        results["vertices"],
        results["edges"],
        results["network"],
        parameters,
        color_by="depth",
        show_vertices=True,
        show_edges=True,
        show_bifurcations=True,
    )
    distribution_fig = _build_share_report_distribution_figure(visualizer, results)

    network_html = to_html(
        network_fig,
        full_html=False,
        include_plotlyjs="inline",
        config={"responsive": True, "displaylogo": False},
    )
    distribution_html = to_html(
        distribution_fig,
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
    )

    core_counts = [
        ("Vertices", len(results["vertices"].get("positions", []))),
        ("Edges", len(results["edges"].get("traces", []))),
        ("Strands", len(results["network"].get("strands", []))),
        ("Bifurcations", len(results["network"].get("bifurcations", []))),
    ]
    headline_metrics = [
        ("Total Length", f"{float(stats.get('total_length', 0.0)):.1f} um"),
        ("Volume Fraction", f"{float(stats.get('volume_fraction', 0.0)):.4f}"),
        ("Mean Radius", f"{float(stats.get('mean_radius', 0.0)):.2f} um"),
        (
            "Bifurcation Density",
            f"{float(stats.get('bifurcation_density', 0.0)):.2f} /mm^3",
        ),
    ]
    parameter_rows = _build_share_report_parameter_rows(parameters)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    dataset_title = escape(dataset_name or "SLAVV dataset")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{dataset_title} - SLAVV Share Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f8fc;
      --panel: #ffffff;
      --line: #d7e3f0;
      --ink: #14324a;
      --muted: #56748c;
      --accent: #1f77b4;
      --accent-soft: #e8f2fb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: linear-gradient(180deg, #edf4fb 0%, var(--bg) 55%, #ffffff 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 28px;
      box-shadow: 0 18px 36px rgba(20, 50, 74, 0.08);
      margin-bottom: 24px;
    }}
    .eyebrow {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 14px 0 10px;
      font-size: 2.1rem;
      line-height: 1.15;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 22px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
    }}
    .label {{
      color: var(--muted);
      font-size: 0.92rem;
      margin-bottom: 6px;
    }}
    .value {{
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--ink);
    }}
    section {{
      margin-top: 24px;
    }}
    h2 {{
      font-size: 1.2rem;
      margin: 0 0 12px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 12px 24px rgba(20, 50, 74, 0.06);
    }}
    .two-up {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    td {{
      padding: 10px 0;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    td:first-child {{
      color: var(--muted);
      width: 52%;
    }}
    footer {{
      margin-top: 24px;
      color: var(--muted);
      font-size: 0.95rem;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <span class="eyebrow">SLAVV Share Report</span>
      <h1>{dataset_title}</h1>
      <p>Generated from the SLAVV processing, curation, visualization, and analysis workflow. This file is self-contained and can be forwarded directly to collaborators for offline review.</p>
      <div class="grid">
        {"".join(f'<div class="card"><div class="label">{escape(label)}</div><div class="value">{escape(str(value))}</div></div>' for label, value in core_counts)}
      </div>
    </section>

    <section>
      <h2>Headline Metrics</h2>
      <div class="grid">
        {"".join(f'<div class="card"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>' for label, value in headline_metrics)}
      </div>
    </section>

    <section>
      <h2>Interactive Network</h2>
      <div class="panel">{network_html}</div>
    </section>

    <section>
      <h2>Distribution Snapshot</h2>
      <div class="panel">{distribution_html}</div>
    </section>

    <section class="two-up">
      <div class="panel">
        <h2>Processing Parameters</h2>
        <table>
          {"".join(f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>" for label, value in parameter_rows)}
        </table>
      </div>
      <div class="panel">
        <h2>Share Context</h2>
        <table>
          <tr><td>Generated at</td><td>{escape(generated_at)}</td></tr>
          <tr><td>Report file</td><td>{escape(make_share_report_filename(dataset_name))}</td></tr>
          <tr><td>Suggested follow-up</td><td>Pair this report with VMV, CASX, or CSV exports when collaborators need deeper inspection.</td></tr>
        </table>
      </div>
    </section>

    <footer>
      Continue in SLAVV to refine curation, compare parameter settings, or export the network in downstream analysis formats.
    </footer>
  </main>
</body>
</html>
"""

    return {
        "html": html,
        "stats": stats,
        "file_name": make_share_report_filename(dataset_name),
        "signature": build_share_report_signature(dataset_name, results, stats),
    }


def _resolve_share_event_log_path(state: Mapping[str, Any] | None = None) -> Path:
    """Resolve the local JSONL log path, allowing tests to override it."""
    custom_path = None if state is None else state.get("share_report_event_log_path")
    return Path(custom_path) if custom_path else DEFAULT_SHARE_REPORT_EVENT_LOG


def record_share_event(
    state: MutableMapping[str, Any],
    event_name: str,
    dataset_name: str,
    report_signature: str,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Append a local share event and mirror simple counters in session state."""
    metrics = dict(state.get("share_report_metrics", {}))
    metrics[event_name] = metrics.get(event_name, 0) + 1
    state["share_report_metrics"] = metrics

    event = {
        "event": event_name,
        "dataset_name": dataset_name,
        "report_signature": report_signature,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        event.update(dict(extra))

    log_path = _resolve_share_event_log_path(state)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")

    return log_path
