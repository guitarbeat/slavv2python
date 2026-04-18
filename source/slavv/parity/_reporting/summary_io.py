"""I/O helpers for summary reporting."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any


def load_report(report_path, logger: logging.Logger) -> dict[str, Any]:
    """Load the comparison report if present."""
    report: dict[str, Any] = {}
    if report_path.exists():
        try:
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
        except Exception as exc:
            logger.error("Failed to load comparison report %s: %s", report_path, exc)
    return report


def infer_date_str(run_name: str) -> str:
    """Infer a display date from the run name."""
    date_str = "Unknown"
    date_part = run_name.split("_")[0]
    if len(date_part) == 8 and date_part.isdigit():
        try:
            date_obj = datetime.strptime(date_part, "%Y%m%d")
            date_str = date_obj.strftime("%Y-%m-%d")
        except Exception:
            pass
    return date_str


def write_summary(output_file, lines: list[str]) -> None:
    """Write the summary file to disk."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated summary: {output_file}")
