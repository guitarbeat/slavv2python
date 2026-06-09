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
