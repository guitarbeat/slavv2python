"""Unit tests for the SLAVV CLI TUI package."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from slavv_python.apps.cli.tui import (
    is_tui_available,
    run_monitor_if_supported,
    run_wizard_if_supported,
)


def test_is_tui_available():
    """Assert is_tui_available returns a boolean."""
    avail = is_tui_available()
    assert isinstance(avail, bool)


@patch("slavv_python.apps.cli.tui._TUI_AVAILABLE", False)
def test_tui_graceful_fallbacks_when_unavailable():
    """Verify that fallbacks work and don't raise errors when TUI is unavailable."""
    assert run_wizard_if_supported() is None
    # Ensure run_monitor_if_supported returns None/fails gracefully
    assert run_monitor_if_supported() is None


@pytest.mark.skipif(not is_tui_available(), reason="TUI extra is not installed")
@patch("slavv_python.apps.cli.tui._TUI_AVAILABLE", True)
@patch("slavv_python.apps.cli.tui.wizard.run_setup_wizard")
def test_run_wizard_if_supported_calls_implementation(mock_run):
    """Verify run_wizard_if_supported invokes implementation when available."""
    mock_run.return_value = {"input_path": "vol.tif", "execute_now": True}
    res = run_wizard_if_supported()
    assert res == {"input_path": "vol.tif", "execute_now": True}
    mock_run.assert_called_once()


@pytest.mark.skipif(not is_tui_available(), reason="TUI extra is not installed")
@patch("slavv_python.apps.cli.tui._TUI_AVAILABLE", True)
@patch("slavv_python.apps.cli.tui.runner_app.SLAVVPipelineApp")
def test_run_monitor_if_supported_runs_app(mock_app_class):
    """Verify run_monitor_if_supported runs the Textual application."""
    mock_app_inst = MagicMock()
    mock_app_class.return_value = mock_app_inst
    run_monitor_if_supported()
    mock_app_inst.run.assert_called_once()


@pytest.mark.skipif(not is_tui_available(), reason="TUI extra is not installed")
@patch("slavv_python.apps.cli.tui._TUI_AVAILABLE", True)
def test_textual_app_reactive_states():
    """Test that SLAVVPipelineApp reactive states and watch methods operate correctly."""
    from slavv_python.apps.cli.tui.runner_app import SLAVVPipelineApp

    app = SLAVVPipelineApp()
    # The app is not running, but we can verify default states
    assert app.current_stage == "Initializing"
    assert app.progress_val == 0.0

    # Ensure update_state runs without crashing when widgets are not yet mounted
    app.update_state("Energy Stage", 25.0, "Computing Hessian matrix...")
    assert app.current_stage == "Energy Stage"
    assert app.progress_val == 25.0
