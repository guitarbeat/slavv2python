"""TUI package for the first-class SLAVV run operations console."""

from __future__ import annotations

import logging

try:
    import textual  # noqa: F401

    _TUI_AVAILABLE = True
except ImportError:
    _TUI_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_tui_available() -> bool:
    """Returns True if the dependencies for the TUI extra are installed."""
    return _TUI_AVAILABLE


def run_monitor_if_supported(run_dir: str | None = None) -> bool | None:
    """Launches the textual pipeline monitor if textual is installed."""
    if not is_tui_available():
        logger.warning("TUI dependencies are not installed. Run 'pip install -e .[tui]' to enable.")
        print("Error: TUI dependency 'textual' is not installed.")
        print('Please run: pip install -e ".[tui]"')
        return None
    from .runner_app import SLAVVPipelineApp

    app = SLAVVPipelineApp(run_dir=run_dir)
    app.run()
    return True
