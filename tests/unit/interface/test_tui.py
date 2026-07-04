"""Unit tests for the SLAVV CLI TUI package."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from slavv_python.interface.cli.tui import (
    is_tui_available,
    run_monitor_if_supported,
)


def test_is_tui_available():
    """Assert is_tui_available returns a boolean."""
    avail = is_tui_available()
    assert isinstance(avail, bool)


@patch("slavv_python.interface.cli.tui._TUI_AVAILABLE", False)
def test_tui_graceful_fallbacks_when_unavailable():
    """Verify monitor dependency errors do not raise when TUI is unavailable."""
    assert run_monitor_if_supported() is None


@pytest.mark.skipif(not is_tui_available(), reason="TUI extra is not installed")
@patch("slavv_python.interface.cli.tui._TUI_AVAILABLE", True)
@patch("slavv_python.interface.cli.tui.runner_app.SLAVVPipelineApp")
def test_run_monitor_if_supported_runs_app(mock_app_class):
    """Verify run_monitor_if_supported runs the Textual application."""
    mock_app_inst = MagicMock()
    mock_app_class.return_value = mock_app_inst
    run_monitor_if_supported()
    mock_app_inst.run.assert_called_once()


@pytest.mark.skipif(not is_tui_available(), reason="TUI extra is not installed")
@patch("slavv_python.interface.cli.tui._TUI_AVAILABLE", True)
def test_textual_app_reactive_states():
    """Test that SLAVVPipelineApp reactive states and watch methods operate correctly."""
    from slavv_python.interface.cli.tui.runner_app import SLAVVPipelineApp

    app = SLAVVPipelineApp()
    # The app is not running, but we can verify default states
    assert app.current_stage == "Initializing"
    assert app.progress_val == 0.0

    # Ensure update_state runs without crashing when widgets are not yet mounted
    app.update_state("Energy Stage", 25.0, "Computing Hessian matrix...")
    assert app.current_stage == "Energy Stage"
    assert app.progress_val == 25.0


@pytest.mark.skipif(not is_tui_available(), reason="TUI extra is not installed")
@patch("slavv_python.interface.cli.tui._TUI_AVAILABLE", True)
def test_energy_bar_update_is_crash_safe_before_mount():
    """Verify _update_energy_bar tolerates unmounted widgets for both states."""
    from slavv_python.interface.cli.monitor_service import EnergyProgress
    from slavv_python.interface.cli.tui.runner_app import SLAVVPipelineApp

    app = SLAVVPipelineApp()
    energy = EnergyProgress(
        units_total=512,
        durable_units_completed=100,
        live_units_completed=140,
        chunks_in_batch=40,
        per_chunk_seconds=1.0,
        eta_seconds=372.0,
        is_live=True,
        log_path=None,
    )
    app._update_energy_bar(energy)
    app._update_energy_bar(None)


@pytest.mark.skipif(not is_tui_available(), reason="TUI extra is not installed")
@patch("slavv_python.interface.cli.tui._TUI_AVAILABLE", True)
def test_app_mounts_and_renders_run(tmp_path):
    """Mount the app headlessly and verify live wiring, bars, and keybindings."""
    import asyncio

    from textual.widgets import ProgressBar

    from slavv_python.interface.cli.tui.runner_app import SLAVVPipelineApp
    from tests.support.run_state_builders import (
        build_snapshot_dict,
        build_stage_snapshot_dict,
        materialize_run_snapshot,
    )

    run_dir = tmp_path / "crop_M_exact"
    energy = build_stage_snapshot_dict(
        "energy", status="running", units_total=512, units_completed=100
    )
    energy["updated_at"] = "2020-01-01T00:00:00Z"
    materialize_run_snapshot(
        run_dir,
        build_snapshot_dict(
            status="running",
            current_stage="energy",
            stages={"energy": energy},
            artifacts={"preprocess_done": "true"},
        ),
    )
    (run_dir / "99_Metadata" / "parity_job.out.log").write_text(
        "[Parallel(n_jobs=6)]: Done 20 tasks | elapsed: 20.0s\n"
        "[Parallel(n_jobs=6)]: Done 40 tasks | elapsed: 40.0s\n",
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = SLAVVPipelineApp(run_dir=run_dir, poll_seconds=100.0)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.current_stage == "energy"
            # 0.05 preprocess + 0.35 * (140/512) live energy fraction.
            assert app.progress_val == pytest.approx(14.57, abs=0.1)
            bar = app.query_one("#energy_bar", ProgressBar)
            assert bar.total == 512
            assert bar.progress == 140
            assert "crop_M_exact" in (app.sub_title or "")
            await pilot.press("p")
            assert "PAUSED" in app.sub_title

    asyncio.run(scenario())
