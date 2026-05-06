"""TUI package providing interactive setup wizards and execution progress monitors."""

from __future__ import annotations

import logging

try:
    import questionary  # noqa: F401
    import textual  # noqa: F401

    _TUI_AVAILABLE = True
except ImportError:
    _TUI_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_tui_available() -> bool:
    """Returns True if the dependencies for the TUI extra are installed."""
    return _TUI_AVAILABLE


def run_wizard_if_supported() -> dict[str, any] | None:
    """Invokes the setup wizard if questionary is installed, otherwise logs a warning."""
    if not is_tui_available():
        logger.warning("TUI dependencies are not installed. Run 'pip install -e .[tui]' to enable.")
        print("Error: TUI dependencies (questionary, textual) are not installed.")
        print('Please run: pip install -e ".[tui]"')
        return None
    from .wizard import run_setup_wizard

    return run_setup_wizard()


def run_monitor_if_supported() -> None:
    """Launches the textual pipeline monitor if textual is installed."""
    if not is_tui_available():
        logger.warning("TUI dependencies are not installed. Run 'pip install -e .[tui]' to enable.")
        print("Error: TUI dependencies (questionary, textual) are not installed.")
        print('Please run: pip install -e ".[tui]"')
        return
    from .runner_app import SLAVVPipelineApp

    app = SLAVVPipelineApp()
    app.run()
