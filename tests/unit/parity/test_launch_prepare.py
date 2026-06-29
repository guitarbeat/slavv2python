from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from slavv_python.analytics.parity.runs.launch_prepare import (
    LaunchPreparationError,
    prepare_detached_exact_run_launch,
    reconcile_run_before_launch,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
def test_reconcile_run_before_launch_rejects_active_writer(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    fake_view = type(
        "FakeView",
        (),
        {
            "effective_status": "running",
            "status_reason": "lease PID 42 is alive.",
            "pid_statuses": [type("Pid", (), {"state": "alive", "pid": 42})()],
        },
    )()

    with (
        patch(
            "slavv_python.interface.cli.monitor_service.load_run_monitor_view",
            return_value=fake_view,
        ),
        pytest.raises(LaunchPreparationError, match="active writer"),
    ):
        reconcile_run_before_launch(run_dir)


@pytest.mark.unit
def test_prepare_detached_exact_run_launch_runs_foreground_probe(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.launch_prepare.reconcile_run_before_launch",
        lambda _root: "interrupted",
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.launch_prepare.run_launch_preflight",
        lambda **_kwargs: {"passed": True},
    )
    probe_calls: list[list[str]] = []

    def _fake_probe(command: list[str], *, cwd: Path) -> int:
        probe_calls.append(command)
        assert cwd.is_dir()
        return 0

    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.launch_prepare.run_foreground_launch_probe",
        _fake_probe,
    )

    detached, foreground = prepare_detached_exact_run_launch(
        dest_run_root=run_dir,
        oracle_root=None,
        dataset_root=None,
        stop_after="energy",
        force_rerun_from="energy",
        memory_safety_fraction=0.8,
        force=False,
        skip_preflight=False,
        skip_foreground_probe=False,
        n_jobs=1,
    )

    assert probe_calls
    assert "--stop-after" in foreground
    assert foreground[foreground.index("--stop-after") + 1] == "preprocess"
    assert detached[detached.index("--stop-after") + 1] == "energy"


@pytest.mark.unit
def test_prepare_detached_exact_run_launch_surfaces_probe_failure(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.launch_prepare.reconcile_run_before_launch",
        lambda _root: "interrupted",
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.launch_prepare.run_launch_preflight",
        lambda **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.runs.launch_prepare.run_foreground_launch_probe",
        lambda *_args, **_kwargs: 1,
    )

    with pytest.raises(LaunchPreparationError, match="Foreground launch probe failed"):
        prepare_detached_exact_run_launch(
            dest_run_root=run_dir,
            oracle_root=None,
            dataset_root=None,
            stop_after="energy",
            force_rerun_from="energy",
            memory_safety_fraction=0.8,
            force=False,
            skip_preflight=True,
            skip_foreground_probe=False,
            n_jobs=1,
        )
