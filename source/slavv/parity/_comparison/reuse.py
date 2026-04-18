"""Reuse and parity-rerun guidance helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _print_reuse_guidance(
    run_root: Path,
    *,
    input_file: str | None = None,
    params: dict[str, Any] | None = None,
    loop_kind: str | None = None,
) -> None:
    """Print reuse eligibility summary using CLI summaries module."""
    from ..cli_summaries import format_reuse_eligibility_summary, generate_reuse_commands
    from ..workflow_assessment import assess_loop_request

    assessment = assess_loop_request(
        run_root,
        loop_kind=loop_kind or "skip_matlab_edges",
        input_path=Path(input_file) if input_file else None,
        params=params,
    )
    if input_file:
        assessment.reuse_commands = generate_reuse_commands(
            assessment,
            run_root=run_root,
            input_file=Path(input_file),
        )
    summary = format_reuse_eligibility_summary(
        assessment,
        run_root=run_root,
        input_file=Path(input_file) if input_file else Path(""),
    )
    print("\n" + summary)


def _resolve_python_parity_import_plan(
    python_parity_rerun_from: str,
) -> tuple[list[str], str]:
    """Return the MATLAB stages needed to seed a Python parity rerun."""
    normalized = str(python_parity_rerun_from or "edges").strip().lower()
    if normalized == "network":
        return ["energy", "vertices", "edges"], "network"
    return ["energy", "vertices"], "edges"
