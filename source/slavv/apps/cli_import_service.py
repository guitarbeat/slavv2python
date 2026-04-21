"""Import-command helpers for the SLAVV CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def build_import_success_lines(
    *,
    checkpoint_dir: str,
    written: Mapping[str, str],
) -> list[str]:
    """Build the printable CLI lines for a successful MATLAB import."""
    lines = [f"Imported {len(written)} stage(s) into {checkpoint_dir}:"]
    lines.extend(f"  {stage}: {path}" for stage, path in written.items())
    lines.extend(
        [
            "",
            "You can now run the Python pipeline with:",
            f"  slavv run -i <image.tif> --checkpoint-dir {checkpoint_dir}",
        ]
    )
    return lines


def build_import_missing_lines() -> list[str]:
    """Build the printable CLI lines for an empty MATLAB import."""
    return ["No MATLAB data files found. Check that the batch folder path is correct."]


__all__ = ["build_import_missing_lines", "build_import_success_lines"]
