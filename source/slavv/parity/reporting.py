"""
Reporting tools for SLAVV comparison.

This module generates summary files and reports for comparison runs.
"""

from __future__ import annotations

from ._reporting.summary_core import generate_summary as _generate_summary
from .matlab_status import load_matlab_status
from .preflight import load_output_preflight
from .run_layout import resolve_run_layout
from .workflow_assessment import load_loop_assessment, load_matlab_health_check


def generate_summary(run_dir, output_file):
    """Generate summary.txt for a comparison run."""
    return _generate_summary(
        run_dir,
        output_file,
        resolve_run_layout=resolve_run_layout,
        load_loop_assessment=load_loop_assessment,
        load_output_preflight=load_output_preflight,
        load_matlab_status=load_matlab_status,
        load_matlab_health_check=load_matlab_health_check,
    )
