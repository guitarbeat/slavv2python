"""Tests for exact-route preflight and unified resume helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from slavv_python.analytics.parity.constants import (
    EXACT_REQUIRED_PARAMETER_VALUES,
    EXPERIMENT_PROVENANCE_PATH,
    RUN_MANIFEST_PATH,
    RUN_SNAPSHOT_PATH,
)
from slavv_python.analytics.parity.params_audit import persist_param_storage
from slavv_python.analytics.parity.preflight import build_exact_preflight_report
from slavv_python.analytics.parity.resume import (
    clear_stale_running_snapshot,
    resolve_exact_run_input_file,
    resolve_exact_run_oracle_root,
)
from slavv_python.engine.constants import STATUS_PENDING, STATUS_RUNNING
from slavv_python.engine.state import RunContext, atomic_write_json, load_json_dict

if TYPE_CHECKING:
    from pathlib import Path


def _write_validated_params(dest: Path) -> None:
    params = dict(EXACT_REQUIRED_PARAMETER_VALUES)
    params["energy_storage_format"] = "npy"
    persist_param_storage(dest, params)


def test_build_exact_preflight_report_passes_with_memory_headroom(tmp_path):
    dest = tmp_path / "run"
    dest.mkdir()
    _write_validated_params(dest)
    atomic_write_json(
        dest / EXPERIMENT_PROVENANCE_PATH,
        {"oracle_size_of_image": [8, 16, 16]},
    )

    with patch(
        "slavv_python.analytics.parity.preflight._available_system_bytes",
        return_value=64 * 1024**3,
    ):
        report = build_exact_preflight_report(
            dest_run_root=dest,
            memory_safety_fraction=0.8,
        )

    assert report["passed"] is True
    assert report["memory_gate"]["passed"] is True


def test_build_exact_preflight_report_fails_without_force_on_low_memory(tmp_path):
    dest = tmp_path / "run"
    dest.mkdir()
    _write_validated_params(dest)
    atomic_write_json(
        dest / EXPERIMENT_PROVENANCE_PATH,
        {"oracle_size_of_image": [128, 512, 512]},
    )

    with patch(
        "slavv_python.analytics.parity.preflight._available_system_bytes",
        return_value=1 * 1024**3,
    ):
        report = build_exact_preflight_report(
            dest_run_root=dest,
            memory_safety_fraction=0.8,
            force=False,
        )

    assert report["passed"] is False
    assert "insufficient_memory" in report["errors"]


def test_build_exact_preflight_report_force_overrides_memory_gate(tmp_path):
    dest = tmp_path / "run"
    dest.mkdir()
    _write_validated_params(dest)
    atomic_write_json(
        dest / EXPERIMENT_PROVENANCE_PATH,
        {"oracle_size_of_image": [128, 512, 512]},
    )

    with patch(
        "slavv_python.analytics.parity.preflight._available_system_bytes",
        return_value=1 * 1024**3,
    ):
        report = build_exact_preflight_report(
            dest_run_root=dest,
            memory_safety_fraction=0.8,
            force=True,
        )

    assert report["passed"] is True
    assert "memory_gate_overridden" in report["warnings"]


def test_resolve_exact_run_input_file_from_snapshot_provenance(tmp_path):
    dataset_tif = tmp_path / "volume.tif"
    dataset_tif.write_bytes(b"0")

    dest = tmp_path / "run"
    dest.mkdir()
    atomic_write_json(
        dest / RUN_SNAPSHOT_PATH,
        {"provenance": {"input_file": str(dataset_tif)}},
    )

    resolved = resolve_exact_run_input_file(dest)
    assert resolved == dataset_tif.resolve()


def test_resolve_exact_run_oracle_root_from_manifest(tmp_path):
    oracle_root = tmp_path / "oracle"
    oracle_root.mkdir()
    dest = tmp_path / "run"
    dest.mkdir()
    atomic_write_json(dest / RUN_MANIFEST_PATH, {"oracle_root": str(oracle_root)})

    assert resolve_exact_run_oracle_root(dest) == oracle_root.resolve()


def test_clear_stale_running_snapshot_sets_pending(tmp_path):
    context = RunContext(run_dir=tmp_path / "run", target_stage="energy")
    context.mark_run_status(STATUS_RUNNING, current_stage="energy", detail="test")
    context.persist()

    assert clear_stale_running_snapshot(context.run_root) is True
    snapshot = load_json_dict(context.snapshot_path) or {}
    assert snapshot.get("status") == STATUS_PENDING
