"""
Reporting tools for SLAVV comparison.

This module generates summary files and reports for comparison runs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from slavv.utils import format_time

from .management import resolve_run_layout

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _report_count(
    report: dict, side: str, top_level_key: str, section: str, nested_key: str
) -> int:
    """Read a count from the preferred top-level field with nested fallbacks."""
    value = report.get(side, {}).get(top_level_key)
    if value is not None:
        return int(value)
    nested = report.get(section, {}).get(nested_key, 0)
    return int(nested)


def generate_summary(run_dir: Path, output_file: Path):
    """Generate summary.txt for a comparison run."""
    layout = resolve_run_layout(run_dir)
    run_root = layout["run_root"]

    # Load comparison report if exists
    report_path = layout["report_file"]
    report = {}
    if report_path.exists():
        try:
            with open(report_path) as f:
                report = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load comparison report {report_path}: {e}")

    # Extract metadata
    run_name = run_root.name

    # Try to infer date from directory name or file modification times
    date_str = "Unknown"
    # Parse YYYYMMDD prefix if present (e.g., 20260228_153045_label)
    date_part = run_name.split("_")[0]
    if len(date_part) == 8 and date_part.isdigit():
        try:
            date_obj = datetime.strptime(date_part, "%Y%m%d")
            date_str = date_obj.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Build summary
    lines = []
    lines.append("=" * 70)
    lines.append("SLAVV Comparison Summary")
    lines.append("=" * 70)
    lines.append(f"Run: {run_name}")
    lines.append(f"Date: {date_str}")
    lines.append("")

    # Performance section
    if report and "performance" in report:
        lines.append("Performance")
        lines.append("-" * 70)
        perf = report["performance"]

        if report.get("matlab", {}).get("elapsed_time"):
            matlab_time = report["matlab"]["elapsed_time"]
            lines.append(f"MATLAB:  {format_time(matlab_time):>12}")

        if report.get("python", {}).get("elapsed_time"):
            python_time = report["python"]["elapsed_time"]
            lines.append(f"Python:  {format_time(python_time):>12}")

        if "speedup" in perf:
            speedup = perf["speedup"]
            faster = perf.get("faster", "Unknown")
            lines.append(f"Speedup: {speedup:>12.2f}x ({faster} faster)")
        lines.append("")

    # Results section
    lines.append("Results")
    lines.append("-" * 70)

    if report:
        # Header
        lines.append(f"{'Component':<15} {'MATLAB':>12} {'Python':>12} {'Difference':>15}")
        lines.append("-" * 70)

        # Vertices
        matlab_verts = _report_count(report, "matlab", "vertices_count", "vertices", "matlab_count")
        python_verts = _report_count(report, "python", "vertices_count", "vertices", "python_count")
        diff_verts = python_verts - matlab_verts
        lines.append(f"{'Vertices':<15} {matlab_verts:>12,} {python_verts:>12,} {diff_verts:>+15,}")

        # Edges
        matlab_edges = _report_count(report, "matlab", "edges_count", "edges", "matlab_count")
        python_edges = _report_count(report, "python", "edges_count", "edges", "python_count")
        diff_edges = python_edges - matlab_edges
        lines.append(f"{'Edges':<15} {matlab_edges:>12,} {python_edges:>12,} {diff_edges:>+15,}")

        # Strands
        matlab_strands = _report_count(
            report, "matlab", "strand_count", "network", "matlab_strand_count"
        )
        python_strands = _report_count(
            report, "python", "network_strands_count", "network", "python_strand_count"
        )
        diff_strands = python_strands - matlab_strands
        lines.append(
            f"{'Strands':<15} {matlab_strands:>12,} {python_strands:>12,} {diff_strands:>+15,}"
        )
    else:
        lines.append("No comparison report found.")

    lines.append("")

    if report and "parity_gate" in report:
        lines.append("Parity Gate")
        lines.append("-" * 70)
        gate = report["parity_gate"]
        lines.append(f"Overall:   {'PASS' if gate.get('passed') else 'FAIL'}")
        lines.append(f"Vertices:  {'PASS' if gate.get('vertices_exact') else 'FAIL'}")
        lines.append(f"Edges:     {'PASS' if gate.get('edges_exact') else 'FAIL'}")
        lines.append(f"Strands:   {'PASS' if gate.get('strands_exact') else 'FAIL'}")

        vertex_mismatch = report.get("vertices", {}).get("matlab_only_samples", [])
        edge_mismatch = report.get("edges", {}).get("matlab_only_samples", [])
        edge_endpoint_mismatch = report.get("edges", {}).get(
            "endpoint_pair_matlab_only_samples", []
        )
        strand_mismatch = report.get("network", {}).get("matlab_only_samples", [])
        if vertex_mismatch:
            lines.append(f"First vertex mismatch: {vertex_mismatch[0]}")
        if edge_endpoint_mismatch:
            lines.append(f"First edge endpoint mismatch: {edge_endpoint_mismatch[0]}")
        if edge_mismatch:
            lines.append(f"First edge mismatch: {edge_mismatch[0]}")
        if strand_mismatch:
            lines.append(f"First strand mismatch: {strand_mismatch[0]}")
        lines.append("")

    edge_diag = report.get("edges", {}).get("diagnostics", {}).get("python", {})
    if edge_diag:
        lines.append("Edge Diagnostics")
        lines.append("-" * 70)
        lines.append(
            f"Candidates: {int(edge_diag.get('candidate_traced_edge_count', 0)):,}"
            f"  Terminal: {int(edge_diag.get('terminal_edge_count', 0)):,}"
            f"  Chosen: {int(edge_diag.get('chosen_edge_count', 0)):,}"
        )
        lines.append(
            f"Dangling: {int(edge_diag.get('dangling_edge_count', 0)):,}"
            f"  Directed duplicates: {int(edge_diag.get('duplicate_directed_pair_count', 0)):,}"
            f"  Antiparallel: {int(edge_diag.get('antiparallel_pair_count', 0)):,}"
        )
        lines.append(
            f"Rejected by energy/conflict/degree/orphan/cycle: "
            f"{int(edge_diag.get('negative_energy_rejected_count', 0)):,}/"
            f"{int(edge_diag.get('conflict_rejected_count', 0)):,}/"
            f"{int(edge_diag.get('degree_pruned_count', 0)):,}/"
            f"{int(edge_diag.get('orphan_pruned_count', 0)):,}/"
            f"{int(edge_diag.get('cycle_pruned_count', 0)):,}"
        )
        lines.append("")

    # Status/notes
    matlab_dir = layout["matlab_dir"]
    python_dir = layout["python_dir"]

    has_matlab = matlab_dir.exists() and any(matlab_dir.iterdir())
    has_python = python_dir.exists() and any(python_dir.iterdir())
    has_plots = layout["plots_dir"].exists()

    lines.append("Status")
    lines.append("-" * 70)
    if has_matlab:
        lines.append("- MATLAB results: Present")
    if has_python:
        lines.append("- Python results: Present")
    if has_plots:
        lines.append("- Visualizations: Present")

    if report:
        matlab_verts = _report_count(report, "matlab", "vertices_count", "vertices", "matlab_count")
        python_verts = _report_count(report, "python", "vertices_count", "vertices", "python_count")

        if matlab_verts == 0 and has_matlab:
            lines.append("- WARNING: MATLAB produced 0 vertices (possible config issue)")
        if python_verts == 0 and has_python:
            lines.append("- WARNING: Python produced 0 vertices (possible config issue)")
        if matlab_verts > 0 and python_verts > 0:
            lines.append("- SUCCESS: Both implementations produced vertices")

    lines.append("")
    lines.append("=" * 70)

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated summary: {output_file}")
