"""Top-level summary generation orchestration."""

from __future__ import annotations

import logging

from .summary_edge_diagnostics import add_edge_diagnostics
from .summary_io import infer_date_str, load_report, write_summary
from .summary_sections import (
    add_header,
    add_parity_gate,
    add_performance,
    add_results,
    add_status,
    add_triage_recommendation,
    add_workflow_sections,
)

logger = logging.getLogger(__name__)


def generate_summary(
    run_dir,
    output_file,
    *,
    resolve_run_layout,
    load_loop_assessment,
    load_output_preflight,
    load_matlab_status,
    load_matlab_health_check,
) -> None:
    """Generate summary.txt for a comparison run."""
    layout = resolve_run_layout(run_dir)
    run_root = layout["run_root"]
    metadata_dir = layout["metadata_dir"]

    report = load_report(layout["report_file"], logger)
    loop_assessment = load_loop_assessment(metadata_dir)
    preflight_report = load_output_preflight(metadata_dir)
    matlab_status = load_matlab_status(metadata_dir)
    matlab_health_check = load_matlab_health_check(metadata_dir)

    run_name = run_root.name
    date_str = infer_date_str(run_name)

    lines: list[str] = []
    add_header(lines, run_name, date_str, report)
    add_workflow_sections(
        lines,
        loop_assessment,
        preflight_report,
        matlab_status,
        matlab_health_check,
    )
    add_performance(lines, report)
    add_results(lines, report)
    add_triage_recommendation(lines, report)
    add_parity_gate(lines, report)
    add_edge_diagnostics(lines, report, layout)
    add_status(lines, report, layout)
    write_summary(output_file, lines)
