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

from .matlab_status import load_matlab_status
from .preflight import load_output_preflight
from .run_layout import resolve_run_layout
from .workflow_assessment import load_loop_assessment, load_matlab_health_check

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
    metadata_dir = layout["metadata_dir"]

    # Load comparison report if exists
    report_path = layout["report_file"]
    report = {}
    if report_path.exists():
        try:
            with open(report_path) as f:
                report = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load comparison report {report_path}: {e}")
    loop_assessment = load_loop_assessment(metadata_dir)
    preflight_report = load_output_preflight(metadata_dir)
    matlab_status = load_matlab_status(metadata_dir)
    matlab_health_check = load_matlab_health_check(metadata_dir)

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
    shared_neighborhood_audit = edge_diagnostics.get("shared_neighborhood_audit", {})
    chosen_candidate_sources = edge_diagnostics.get("chosen_candidate_sources", {})
    extra_frontier_overlap = edge_diagnostics.get("extra_frontier_missing_vertex_overlap", {})
    if edge_diag or candidate_coverage or candidate_audit or shared_neighborhood_audit:
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
            if conflict_rejected_by_source := edge_diag.get("conflict_rejected_by_source", {}):
                lines.append(
                    "Conflict rejects by source frontier/watershed/fallback/unknown: "
                    f"{int(conflict_rejected_by_source.get('frontier', 0)):,}/"
                    f"{int(conflict_rejected_by_source.get('watershed', 0)):,}/"
                    f"{int(conflict_rejected_by_source.get('fallback', 0)):,}/"
                    f"{int(conflict_rejected_by_source.get('unknown', 0)):,}"
                )
            if conflict_blocking_source_counts := edge_diag.get(
                "conflict_blocking_source_counts", {}
            ):
                lines.append(
                    "Conflict blockers by source frontier/watershed/fallback/unknown: "
                    f"{int(conflict_blocking_source_counts.get('frontier', 0)):,}/"
                    f"{int(conflict_blocking_source_counts.get('watershed', 0)):,}/"
                    f"{int(conflict_blocking_source_counts.get('fallback', 0)):,}/"
                    f"{int(conflict_blocking_source_counts.get('unknown', 0)):,}"
                )
            if conflict_source_pairs := edge_diag.get("conflict_source_pairs", {}):
                lines.append(
                    "Conflict source pairs f->f/f->w/w->f/w->w: "
                    f"{int(conflict_source_pairs.get('frontier->frontier', 0)):,}/"
                    f"{int(conflict_source_pairs.get('frontier->watershed', 0)):,}/"
                    f"{int(conflict_source_pairs.get('watershed->frontier', 0)):,}/"
                    f"{int(conflict_source_pairs.get('watershed->watershed', 0)):,}"
                )
        final_matched_endpoint_pairs = int(
            report.get("edges", {}).get("matched_endpoint_pair_count", 0)
        )
        final_missing_endpoint_pairs = int(
            report.get("edges", {}).get("missing_endpoint_pair_count", 0)
        )
        final_extra_endpoint_pairs = int(
            report.get("edges", {}).get("extra_endpoint_pair_count", 0)
        )
        if (
            final_matched_endpoint_pairs
            or final_missing_endpoint_pairs
            or final_extra_endpoint_pairs
        ):
            lines.append(
                "Final endpoint pairs matched/matlab-only/python-only: "
                f"{final_matched_endpoint_pairs:,}/"
                f"{final_missing_endpoint_pairs:,}/"
                f"{final_extra_endpoint_pairs:,}"
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
            if any(
                key in candidate_coverage
                for key in (
                    "frontier_only_candidate_endpoint_pair_count",
                    "watershed_only_candidate_endpoint_pair_count",
                    "fallback_only_candidate_endpoint_pair_count",
                )
            ):
                lines.append(
                    "Candidate endpoint pair provenance frontier-only/watershed-only/fallback-only: "
                    f"{int(candidate_coverage.get('frontier_only_candidate_endpoint_pair_count', 0)):,}/"
                    f"{int(candidate_coverage.get('watershed_only_candidate_endpoint_pair_count', 0)):,}/"
                    f"{int(candidate_coverage.get('fallback_only_candidate_endpoint_pair_count', 0)):,}"
                )
            if missing_candidate_pairs := candidate_coverage.get(
                "missing_matlab_endpoint_pair_samples", []
            ):
                lines.append(f"First missing candidate endpoint pair: {missing_candidate_pairs[0]}")
            if missing_seed_origins := candidate_coverage.get(
                "missing_matlab_seed_origin_samples", []
            ):
                top_seed_origin = missing_seed_origins[0]
                lines.append(
                    "Top missing seed origin "
                    f"{int(top_seed_origin.get('seed_origin_index', -1))}: "
                    f"missing matlab incident pairs "
                    f"{int(top_seed_origin.get('missing_matlab_incident_endpoint_pair_count', 0)):,}"
                    f"  seed candidate pairs "
                    f"{int(top_seed_origin.get('candidate_endpoint_pair_count', 0)):,}"
                )
                if missing_seed_pairs := top_seed_origin.get(
                    "missing_matlab_incident_endpoint_pair_samples", []
                ):
                    lines.append(f"First missing pair at top seed origin: {missing_seed_pairs[0]}")
            if supplement_candidate_pairs := candidate_coverage.get(
                "supplement_candidate_endpoint_pair_samples", []
            ):
                lines.append(
                    f"First watershed supplement candidate pair: {supplement_candidate_pairs[0]}"
                )
            if extra_seed_origins := candidate_coverage.get(
                "extra_candidate_seed_origin_samples", []
            ):
                top_extra_origin = extra_seed_origins[0]
                lines.append(
                    "Top extra seed origin "
                    f"{int(top_extra_origin.get('seed_origin_index', -1))}: "
                    f"extra candidate pairs "
                    f"{int(top_extra_origin.get('extra_candidate_endpoint_pair_count', 0)):,}"
                    f"  seed candidate pairs "
                    f"{int(top_extra_origin.get('candidate_endpoint_pair_count', 0)):,}"
                )
                if extra_seed_pairs := top_extra_origin.get(
                    "extra_candidate_endpoint_pair_samples", []
                ):
                    lines.append(f"First extra pair at top seed origin: {extra_seed_pairs[0]}")
        if chosen_candidate_sources:
            chosen_counts = chosen_candidate_sources.get("counts", {})
            lines.append(
                "Chosen candidate sources frontier/watershed/geodesic/fallback: "
                f"{int(chosen_counts.get('frontier', 0)):,}/"
                f"{int(chosen_counts.get('watershed', 0)):,}/"
                f"{int(chosen_counts.get('geodesic', 0)):,}/"
                f"{int(chosen_counts.get('fallback', 0)):,}"
            )
            watershed_pair_count = int(
                chosen_candidate_sources.get("watershed_endpoint_pair_count", 0)
            )
            if watershed_pair_count > 0:
                lines.append(
                    "Chosen watershed endpoint pairs total/matched-matlab/extra-python: "
                    f"{watershed_pair_count:,}/"
                    f"{int(chosen_candidate_sources.get('watershed_matched_matlab_endpoint_pair_count', 0)):,}/"
                    f"{int(chosen_candidate_sources.get('watershed_extra_python_endpoint_pair_count', 0)):,}"
                )
            geodesic_pair_count = int(
                chosen_candidate_sources.get("geodesic_endpoint_pair_count", 0)
            )
            if geodesic_pair_count > 0:
                lines.append(
                    "Chosen geodesic endpoint pairs total/matched-matlab/extra-python: "
                    f"{geodesic_pair_count:,}/"
                    f"{int(chosen_candidate_sources.get('geodesic_matched_matlab_endpoint_pair_count', 0)):,}/"
                    f"{int(chosen_candidate_sources.get('geodesic_extra_python_endpoint_pair_count', 0)):,}"
                )
            source_breakdown = chosen_candidate_sources.get("source_breakdown", {})
            for source_label in ("frontier", "watershed", "geodesic", "fallback"):
                source_summary = source_breakdown.get(source_label, {})
                if not source_summary:
                    continue
                matched = int(source_summary.get("matched_matlab_edge_count", 0))
                extra = int(source_summary.get("extra_python_edge_count", 0))
                matched_stats = source_summary.get("matched", {})
                extra_stats = source_summary.get("extra", {})
                if matched == 0 and extra == 0:
                    continue
                lines.append(f"Chosen {source_label} edges matched/extra: {matched:,}/{extra:,}")
                if matched_stats or extra_stats:
                    matched_energy = matched_stats.get("median_energy")
                    extra_energy = extra_stats.get("median_energy")
                    matched_length = matched_stats.get("median_length")
                    extra_length = extra_stats.get("median_length")
                    fragments = []
                    if matched_energy is not None or extra_energy is not None:
                        matched_energy_text = (
                            f"{float(matched_energy):.1f}" if matched_energy is not None else "n/a"
                        )
                        extra_energy_text = (
                            f"{float(extra_energy):.1f}" if extra_energy is not None else "n/a"
                        )
                        fragments.append(
                            f"median energy matched/extra {matched_energy_text}/{extra_energy_text}"
                        )
                    if matched_length is not None or extra_length is not None:
                        matched_length_text = (
                            f"{round(float(matched_length))}"
                            if matched_length is not None
                            else "n/a"
                        )
                        extra_length_text = (
                            f"{round(float(extra_length))}" if extra_length is not None else "n/a"
                        )
                        fragments.append(
                            f"median length matched/extra {matched_length_text}/{extra_length_text}"
                        )
                    if fragments:
                        lines.append(f"Chosen {source_label} profile: " + "  ".join(fragments))
        if extra_frontier_overlap:
            lines.append(
                "Extra frontier edges sharing missing-matlab vertex total: "
                f"{int(extra_frontier_overlap.get('shared_missing_vertex_edge_count', 0)):,}/"
                f"{int(extra_frontier_overlap.get('extra_frontier_edge_count', 0)):,}"
            )
            if strength_overlap := extra_frontier_overlap.get("top_strength_overlap_counts", {}):
                segments = []
                for threshold in ("20", "50", "100"):
                    entry = strength_overlap.get(threshold, {})
                    if not entry:
                        continue
                    segments.append(
                        f"top{int(entry.get('threshold', 0))}:"
                        f"{int(entry.get('shared_missing_vertex_count', 0))}/"
                        f"{int(entry.get('evaluated_edge_count', 0))}"
                    )
                if segments:
                    lines.append(
                        "Strongest extra frontier sharing missing-matlab vertex: "
                        + "  ".join(segments)
                    )
            if top_shared_vertices := extra_frontier_overlap.get("top_shared_vertices", []):
                formatted_vertices = []
                for entry in top_shared_vertices[:3]:
                    formatted_vertices.append(
                        f"{int(entry.get('vertex_index', -1))}"
                        f"(m{int(entry.get('missing_matlab_endpoint_pair_count', 0))}/"
                        f"e{int(entry.get('extra_frontier_endpoint_pair_count', 0))})"
                    )
                if formatted_vertices:
                    lines.append(
                        "Top shared frontier/missing vertices: " + " ".join(formatted_vertices)
                    )
                candidate_hits = []
                for entry in top_shared_vertices[:3]:
                    candidate_hits.append(
                        f"{int(entry.get('vertex_index', -1))}"
                        f"({int(entry.get('missing_matlab_pairs_present_in_candidates', 0))}/"
                        f"{int(entry.get('missing_matlab_endpoint_pair_count', 0))})"
                    )
                if candidate_hits:
                    lines.append(
                        "Top shared vertex missing-pair candidate hits: " + " ".join(candidate_hits)
                    )
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
            if audit_top := candidate_audit.get("top_origin_summaries", []):
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
            if pair_source_breakdown := candidate_audit.get("pair_source_breakdown", {}):
                lines.append(
                    "Candidate audit pair provenance frontier-only/watershed-only/fallback-only/multi-source: "
                    f"{int(pair_source_breakdown.get('frontier_only_pair_count', 0)):,}/"
                    f"{int(pair_source_breakdown.get('watershed_only_pair_count', 0)):,}/"
                    f"{int(pair_source_breakdown.get('fallback_only_pair_count', 0)):,}/"
                    f"{int(pair_source_breakdown.get('multi_source_pair_count', 0)):,}"
                )
                if watershed_only_pairs := pair_source_breakdown.get(
                    "watershed_only_endpoint_pair_samples", []
                ):
                    lines.append(f"First watershed-only candidate pair: {watershed_only_pairs[0]}")
            if audit_diag := candidate_audit.get("diagnostic_counters", {}):
                lines.append(
                    "Audit rejections reachability/mutual/endpoint-degree/energy/metric-threshold/cap/short/accepted: "
                    f"{int(audit_diag.get('watershed_reachability_rejected', 0)):,}/"
                    f"{int(audit_diag.get('watershed_mutual_frontier_rejected', 0)):,}/"
                    f"{int(audit_diag.get('watershed_endpoint_degree_rejected', 0)):,}/"
                    f"{int(audit_diag.get('watershed_energy_rejected', 0)):,}/"
                    f"{int(audit_diag.get('watershed_metric_threshold_rejected', 0)):,}/"
                    f"{int(audit_diag.get('watershed_cap_rejected', 0)):,}/"
                    f"{int(audit_diag.get('watershed_short_trace_rejected', 0)):,}/"
                    f"{int(audit_diag.get('watershed_accepted', 0)):,}"
                )
        if shared_neighborhood_audit:
            shared_neighborhood_audit_path = (
                layout["analysis_dir"] / "shared_neighborhood_audit.json"
            )
            lines.append(f"Shared neighborhood audit artifact: {shared_neighborhood_audit_path}")
            top_neighborhood = shared_neighborhood_audit.get("top_neighborhood", {})
            if isinstance(top_neighborhood, dict) and top_neighborhood:
                lines.append(
                    "Top shared neighborhood "
                    f"{int(top_neighborhood.get('origin_index', -1))}: "
                    f"missing/candidate/final "
                    f"{int(top_neighborhood.get('missing_matlab_incident_endpoint_pair_count', 0)):,}/"
                    f"{int(top_neighborhood.get('candidate_endpoint_pair_count', 0)):,}/"
                    f"{int(top_neighborhood.get('final_chosen_endpoint_pair_count', 0)):,}"
                )
                lines.append(
                    "First divergence point: "
                    f"{top_neighborhood.get('first_divergence_stage', 'unknown')} - "
                    f"{top_neighborhood.get('first_divergence_reason', 'unknown')}"
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
        if stop_reason_counts := edge_diag.get("stop_reason_counts", {}):
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

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated summary: {output_file}")
