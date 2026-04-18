"""I/O helpers for summary reporting."""

from __future__ import annotations

from typing import Any

from .._persistence import (
    infer_date_str as _infer_date_str,
)
from .._persistence import (
    load_json_dict_or_empty,
    write_lines_file,
)


def load_report(report_path, logger: Any) -> dict[str, Any]:
    """Load the comparison report if present."""
    report = load_json_dict_or_empty(report_path)
    if report_path.exists() and not report:
        logger.error("Failed to load comparison report %s", report_path)
    return report


def infer_date_str(run_name: str) -> str:
    """Infer a display date from the run name."""
    return _infer_date_str(run_name)


def write_summary(output_file, lines: list[str]) -> None:
    """Write the summary file to disk."""
    write_lines_file(output_file, lines)
    print(f"Generated summary: {output_file}")
