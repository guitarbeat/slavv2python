"""Tests for CLI info helpers."""

from __future__ import annotations

from source.apps.cli_info_service import load_info_lines


def test_load_info_lines_delegates_to_report_format():
    lines = load_info_lines(
        version="9.9.9",
        system_info={"Python": "3.12.1", "Platform": "Windows"},
    )

    assert lines == [
        "slavv 9.9.9",
        "",
        "  Python: 3.12.1",
        "  Platform: Windows",
    ]
