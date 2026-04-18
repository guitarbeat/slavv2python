"""Section builders for comparison summaries."""

from __future__ import annotations

from typing import Any

from slavv.utils import format_time


def _report_count(
    report: dict[str, Any], side: str, top_level_key: str, section: str, nested_key: str
) -> int:
    """Read a count from the preferred top-level field with nested fallbacks."""
    value = report.get(side, {}).get(top_level_key)
    if value is not None:
        return int(value)
    nested = report.get(section, {}).get(nested_key, 0)
    return int(nested)


def add_header(lines: list[str], run_name: str, date_str: str, report: dict[str, Any]) -> None:
    """Append the summary header."""
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


def add_workflow_sections(
    lines: list[str],
    loop_assessment: dict[str, Any],
    preflight_report: dict[str, Any],
    matlab_status: dict[str, Any],
    matlab_health_check: dict[str, Any],
) -> None:
    """Append workflow and health sections."""
    if loop_assessment:
        lines.append("Workflow Decision")
        lines.append("-" * 70)
        lines.append(
            "Verdict: " + str(loop_assessment.get("verdict", "unknown")).replace("_", " ").upper()
        )
        lines.append(f"Safe to reuse: {bool(loop_assessment.get('safe_to_reuse', False))}")
        lines.append(
            f"Safe to analyze only: {bool(loop_assessment.get('safe_to_analyze_only', False))}"
        )
        lines.append(
            f"Requires fresh MATLAB: {bool(loop_assessment.get('requires_fresh_matlab', False))}"
        )
        if recommended_action := str(loop_assessment.get("recommended_action", "") or "").strip():
            lines.append(f"Recommended action: {recommended_action}")
        lines.append("Assessment artifact: 99_Metadata/loop_assessment.json")
        lines.append("")

    if preflight_report:
        lines.append("Preflight Status")
        lines.append("-" * 70)
        lines.append(f"Output preflight: {preflight_report.get('preflight_status', 'unknown')}")
        if recommended_action := str(preflight_report.get("recommended_action", "") or "").strip():
            lines.append(f"Preflight action: {recommended_action}")
        lines.append("")

    if matlab_status:
        lines.append("MATLAB Status")
        lines.append("-" * 70)
        lines.append(f"MATLAB resume mode: {matlab_status.get('matlab_resume_mode', 'unknown')}")
        lines.append(
            f"MATLAB rerun prediction: {matlab_status.get('matlab_rerun_prediction', 'unknown')}"
        )
        lines.append("")

    if matlab_health_check:
        lines.append("MATLAB Health Check")
        lines.append("-" * 70)
        lines.append(f"Health check passed: {bool(matlab_health_check.get('success', False))}")
        if message := str(matlab_health_check.get("message", "") or "").strip():
            lines.append(message)
        lines.append("")


def add_performance(lines: list[str], report: dict[str, Any]) -> None:
    """Append the performance section."""
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


def add_results(lines: list[str], report: dict[str, Any]) -> None:
    """Append the results section."""
    lines.append("Results")
    lines.append("-" * 70)

    if report:
        lines.append(f"{'Component':<15} {'MATLAB':>12} {'Python':>12} {'Difference':>15}")
        lines.append("-" * 70)

        matlab_verts = _report_count(report, "matlab", "vertices_count", "vertices", "matlab_count")
        python_verts = _report_count(report, "python", "vertices_count", "vertices", "python_count")
        diff_verts = python_verts - matlab_verts
        lines.append(f"{'Vertices':<15} {matlab_verts:>12,} {python_verts:>12,} {diff_verts:>+15,}")

        matlab_edges = _report_count(report, "matlab", "edges_count", "edges", "matlab_count")
        python_edges = _report_count(report, "python", "edges_count", "edges", "python_count")
        diff_edges = python_edges - matlab_edges
        lines.append(f"{'Edges':<15} {matlab_edges:>12,} {python_edges:>12,} {diff_edges:>+15,}")

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


def add_triage_recommendation(lines: list[str], report: dict[str, Any]) -> None:
    """Append the triage recommendation section."""
    candidate_coverage = (
        report.get("edges", {}).get("diagnostics", {}).get("candidate_endpoint_coverage", {})
    )
    if report:
        missing_candidate_pairs = int(
            candidate_coverage.get("missing_matlab_endpoint_pair_count", 0)
        )
        extra_candidate_pairs = int(
            candidate_coverage.get("extra_candidate_endpoint_pair_count", 0)
        )
        lines.append("Triage Recommendation")
        lines.append("-" * 70)
        if missing_candidate_pairs > 0:
            lines.append("Start with candidate-endpoint coverage before edge or strand diffs.")
            lines.append(
                "Focus: missing MATLAB endpoint pairs indicate the parity gap is still in candidate generation."
            )
        elif extra_candidate_pairs > 0:
            lines.append("Start with candidate-endpoint coverage before final edge diffs.")
            lines.append(
                "Focus: extra candidate endpoint pairs suggest Python is over-generating before cleanup."
            )
        else:
            lines.append(
                "Candidate-endpoint coverage looks healthy; move to final edge and strand diffs next."
            )
        lines.append("")


def add_parity_gate(lines: list[str], report: dict[str, Any]) -> None:
    """Append the parity gate section."""
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


def add_status(lines: list[str], report: dict[str, Any], layout: dict[str, Any]) -> None:
    """Append the status section."""
    matlab_dir = layout["matlab_dir"]
    python_dir = layout["python_dir"]

    has_matlab = matlab_dir.exists() and any(matlab_dir.iterdir())
    has_python = python_dir.exists() and any(python_dir.iterdir())
    has_plots = layout["plots_dir"].exists()
    quick_json = layout["analysis_dir"] / "comparison_quick_view.json"
    quick_tsv = layout["analysis_dir"] / "comparison_quick_view.tsv"

    lines.append("Status")
    lines.append("-" * 70)
    if has_matlab:
        lines.append("- MATLAB results: Present")
    if has_python:
        lines.append("- Python results: Present")
    if has_plots:
        lines.append("- Visualizations: Present")
    if quick_json.exists():
        lines.append(f"- Quick compare JSON: {quick_json}")
    if quick_tsv.exists():
        lines.append(f"- Quick compare TSV: {quick_tsv}")

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
