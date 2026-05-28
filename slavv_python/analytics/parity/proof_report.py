"""Human-readable exact-route proof report rendering."""

from __future__ import annotations

from typing import Any

def render_exact_proof_report(report: dict[str, Any]) -> str:
    """Render a human-readable exact-proof report."""
    stage_lines = [
        f"{stage}: {'PASS' if summary.get('passed') else 'FAIL'}"
        for stage, summary in report["stage_summaries"].items()
    ]
    lines = [
        "Exact proof report",
        f"Status: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Stages: {', '.join(report.get('stages', []))}",
    ]
    report_scope = report.get("report_scope")
    if isinstance(report_scope, str) and report_scope:
        lines.append(f"Scope: {report_scope}")
    lines.extend(
        [
            "",
            "Stage summary",
            *stage_lines,
        ]
    )

    candidate_surface = report.get("candidate_surface")
    if isinstance(candidate_surface, dict):
        lines.extend(
            [
                "",
                "Candidate surface",
                (
                    "Counts: "
                    f"MATLAB={candidate_surface.get('matlab_pair_count', 0)} "
                    f"Python={candidate_surface.get('python_pair_count', 0)} "
                    f"matched={candidate_surface.get('matched_pair_count', 0)} "
                    f"missing={candidate_surface.get('missing_pair_count', 0)} "
                    f"extra={candidate_surface.get('extra_pair_count', 0)}"
                ),
            ]
        )

    first_failure = report.get("first_failure")
    if isinstance(first_failure, dict):
        lines.extend(
            [
                "",
                "First failure",
                f"Stage: {first_failure['stage']}",
                f"Field: {first_failure['field_path']}",
                f"Mismatch: {first_failure['mismatch_type']}",
                f"MATLAB: {first_failure['matlab_preview']}",
                f"Python: {first_failure['python_preview']}",
            ]
        )

    return "\n".join(lines)


__all__ = ["render_exact_proof_report"]
