"""Tests for output-root preflight decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import slavv.evaluation.preflight as preflight_module
from slavv.evaluation.preflight import (
    PREFLIGHT_STATUS_BLOCKED,
    PREFLIGHT_STATUS_PASSED,
    PREFLIGHT_STATUS_WARNING,
    evaluate_output_root_preflight,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_preflight_accepts_writable_local_output_root(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "comparison_run"

    monkeypatch.setattr(preflight_module, "_is_onedrive_suspected", lambda _path: False)
    monkeypatch.setattr(preflight_module, "_is_non_local_path_suspected", lambda *_args: False)
    monkeypatch.setattr(preflight_module, "measure_free_space_gb", lambda _path: 32.0)

    report = evaluate_output_root_preflight(output_root)

    assert report.preflight_status == PREFLIGHT_STATUS_PASSED
    assert report.allows_launch is True
    assert report.writable is True
    assert report.errors == []
    assert report.warnings == []


def test_preflight_blocks_low_free_space(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "comparison_run"

    monkeypatch.setattr(preflight_module, "_is_onedrive_suspected", lambda _path: False)
    monkeypatch.setattr(preflight_module, "_is_non_local_path_suspected", lambda *_args: False)
    monkeypatch.setattr(preflight_module, "measure_free_space_gb", lambda _path: 1.5)

    report = evaluate_output_root_preflight(output_root, required_free_space_gb=5.0)

    assert report.preflight_status == PREFLIGHT_STATUS_BLOCKED
    assert report.allows_launch is False
    assert any("Low disk space" in error for error in report.errors)


def test_preflight_blocks_unwritable_output_root(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "comparison_run"

    monkeypatch.setattr(preflight_module, "_is_onedrive_suspected", lambda _path: False)
    monkeypatch.setattr(preflight_module, "_is_non_local_path_suspected", lambda *_args: False)
    monkeypatch.setattr(preflight_module, "measure_free_space_gb", lambda _path: 32.0)

    def _raise_not_writable(_directory: Path) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(preflight_module, "_probe_directory_writable", _raise_not_writable)

    report = evaluate_output_root_preflight(output_root)

    assert report.preflight_status == PREFLIGHT_STATUS_BLOCKED
    assert report.allows_launch is False
    assert any("permission denied" in error for error in report.errors)


def test_preflight_warns_for_onedrive_suspected_root(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "comparison_run"

    monkeypatch.setattr(preflight_module, "_is_onedrive_suspected", lambda _path: True)
    monkeypatch.setattr(preflight_module, "_is_non_local_path_suspected", lambda *_args: False)
    monkeypatch.setattr(preflight_module, "measure_free_space_gb", lambda _path: 32.0)

    report = evaluate_output_root_preflight(output_root)

    assert report.preflight_status == PREFLIGHT_STATUS_WARNING
    assert report.allows_launch is True
    assert report.onedrive_suspected is True
    assert any("OneDrive" in warning for warning in report.warnings)
