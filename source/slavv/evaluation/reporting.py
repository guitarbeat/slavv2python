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
    comparison_mode = report.get("python", {}).get("comparison_mode", {})
    if comparison_mode.get("energy_source"):
        lines.append(f"Python energy source: {comparison_mode['energy_source']}")
    if comparison_mode.get("result_source"):
        lines.append(f"Python result source: {comparison_mode['result_source']}")
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

    edge_diagnostics = report.get("edges", {}).get("diagnostics", {})
    edge_diag = edge_diagnostics.get("python", {})
    candidate_coverage = edge_diagnostics.get("candidate_endpoint_coverage", {})
    candidate_audit = edge_diagnostics.get("candidate_audit", {})
    if edge_diag or candidate_coverage or candidate_audit:
        lines.append("Edge Diagnostics")
        lines.append("-" * 70)
        if edge_diag:
            lines.append(
                f"Candidates: {int(edge_diag.get('candidate_traced_edge_count', 0)):,}"
                f"  Terminal: {int(edge_diag.get('terminal_edge_count', 0)):,}"
                f"  Chosen: {int(edge_diag.get('chosen_edge_count', 0)):,}"
            )
            if "watershed_join_supplement_count" in edge_diag:
                lines.append(
                    "Watershed join supplements: "
                    f"{int(edge_diag.get('watershed_join_supplement_count', 0)):,}"
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
        if candidate_coverage:
            lines.append(
                "Candidate endpoint pairs candidate/matched-matlab/missing-matlab: "
                f"{int(candidate_coverage.get('candidate_endpoint_pair_count', 0)):,}/"
                f"{int(candidate_coverage.get('matched_matlab_endpoint_pair_count', 0)):,}/"
                f"{int(candidate_coverage.get('missing_matlab_endpoint_pair_count', 0)):,}"
            )
            lines.append(
                "Candidate endpoint pairs extra-candidate/final-python: "
                f"{int(candidate_coverage.get('extra_candidate_endpoint_pair_count', 0)):,}/"
                f"{int(candidate_coverage.get('python_endpoint_pair_count', 0)):,}"
            )
            missing_candidate_pairs = candidate_coverage.get(
                "missing_matlab_endpoint_pair_samples", []
            )
            if missing_candidate_pairs:
                lines.append(f"First missing candidate endpoint pair: {missing_candidate_pairs[0]}")
            missing_seed_origins = candidate_coverage.get("missing_matlab_seed_origin_samples", [])
            if missing_seed_origins:
                top_seed_origin = missing_seed_origins[0]
                lines.append(
                    "Top missing seed origin "
                    f"{int(top_seed_origin.get('seed_origin_index', -1))}: "
                    f"missing matlab incident pairs "
                    f"{int(top_seed_origin.get('missing_matlab_incident_endpoint_pair_count', 0)):,}"
                    f"  seed candidate pairs "
                    f"{int(top_seed_origin.get('candidate_endpoint_pair_count', 0)):,}"
                )
                missing_seed_pairs = top_seed_origin.get(
                    "missing_matlab_incident_endpoint_pair_samples", []
                )
                if missing_seed_pairs:
                    lines.append(f"First missing pair at top seed origin: {missing_seed_pairs[0]}")
        if candidate_audit:
            lines.append(
                "Candidate audit: "
                f"schema=v{int(candidate_audit.get('schema_version', 1))} "
                f"frontier={int(candidate_audit.get('source_breakdown', {}).get('frontier', {}).get('candidate_connection_count', 0)):,}/"
                f"watershed={int(candidate_audit.get('source_breakdown', {}).get('watershed', {}).get('candidate_connection_count', 0)):,}/"
                f"fallback={int(candidate_audit.get('source_breakdown', {}).get('fallback', {}).get('candidate_connection_count', 0)):,}"
            )
            candidate_audit_path = (
                layout["python_dir"] / "stages" / "edges" / "candidate_audit.json"
            )
            candidate_manifest_path = layout["python_dir"] / "stages" / "edges" / "candidates.pkl"
            lines.append(f"Candidate audit artifact: {candidate_audit_path}")
            lines.append(f"Candidate manifest path: {candidate_manifest_path}")
            audit_top = candidate_audit.get("top_origin_summaries", [])
            if audit_top:
                lines.append("Top audit origin summaries (watershed/frontier/fallback total):")
                for entry in audit_top[:3]:
                    if not isinstance(entry, dict):
                        continue
                    lines.append(
                        f"  Origin {int(entry.get('origin_index', -1))}: "
                        f"w={int(entry.get('watershed_candidate_count', 0)):,}/"
                        f"f={int(entry.get('frontier_candidate_count', 0)):,}/"
                        f"x={int(entry.get('fallback_candidate_count', 0)):,} "
                        f"total={int(entry.get('candidate_connection_count', 0)):,}"
                    )
            audit_diag = candidate_audit.get("diagnostic_counters", {})
            if audit_diag:
                lines.append(
                    "Audit rejections reachability/energy/cap/short/accepted: "
                    f"{int(audit_diag.get('watershed_reachability_rejected', 0)):,}/"
                    f"{int(audit_diag.get('watershed_energy_rejected', 0)):,}/"
                    f"{int(audit_diag.get('watershed_cap_rejected', 0)):,}/"
                    f"{int(audit_diag.get('watershed_short_trace_rejected', 0)):,}/"
                    f"{int(audit_diag.get('watershed_accepted', 0)):,}"
                )
        if any(
            key in edge_diag
            for key in (
                "terminal_direct_hit_count",
                "terminal_reverse_center_hit_count",
                "terminal_reverse_near_hit_count",
            )
        ):
            lines.append(
                "Terminal resolution direct/reverse-center/reverse-near: "
                f"{int(edge_diag.get('terminal_direct_hit_count', 0)):,}/"
                f"{int(edge_diag.get('terminal_reverse_center_hit_count', 0)):,}/"
                f"{int(edge_diag.get('terminal_reverse_near_hit_count', 0)):,}"
            )
        stop_reason_counts = edge_diag.get("stop_reason_counts", {})
        if stop_reason_counts:
            lines.append(
                "Stop reasons bounds/nan/threshold/rise/max-steps/direct-hit: "
                f"{int(stop_reason_counts.get('bounds', 0)):,}/"
                f"{int(stop_reason_counts.get('nan', 0)):,}/"
                f"{int(stop_reason_counts.get('energy_threshold', 0)):,}/"
                f"{int(stop_reason_counts.get('energy_rise_step_halving', 0)):,}/"
                f"{int(stop_reason_counts.get('max_steps', 0)):,}/"
                f"{int(stop_reason_counts.get('direct_terminal_hit', 0)):,}"
            )
            if any(
                key in stop_reason_counts
                for key in (
                    "frontier_exhausted_nonnegative",
                    "length_limit",
                    "terminal_frontier_hit",
                )
            ):
                lines.append(
                    "Frontier stop reasons exhausted/length-limit/terminal-hit: "
                    f"{int(stop_reason_counts.get('frontier_exhausted_nonnegative', 0)):,}/"
                    f"{int(stop_reason_counts.get('length_limit', 0)):,}/"
                    f"{int(stop_reason_counts.get('terminal_frontier_hit', 0)):,}"
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
