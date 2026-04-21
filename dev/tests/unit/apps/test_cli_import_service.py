"""Tests for CLI import-command helpers."""

from __future__ import annotations

from slavv.apps.cli_import_service import build_import_missing_lines, build_import_success_lines


def test_build_import_success_lines_lists_written_stages():
    lines = build_import_success_lines(
        checkpoint_dir="checkpoints",
        written={"energy": "a.pkl", "vertices": "b.pkl"},
    )

    assert lines == [
        "Imported 2 stage(s) into checkpoints:",
        "  energy: a.pkl",
        "  vertices: b.pkl",
        "",
        "You can now run the Python pipeline with:",
        "  slavv run -i <image.tif> --checkpoint-dir checkpoints",
    ]


def test_build_import_missing_lines_returns_single_message():
    assert build_import_missing_lines() == [
        "No MATLAB data files found. Check that the batch folder path is correct."
    ]
