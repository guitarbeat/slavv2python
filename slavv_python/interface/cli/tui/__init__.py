"""TUI package providing interactive setup wizards and execution progress monitors."""

from __future__ import annotations

import logging
from typing import Any

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


def run_wizard_if_supported() -> dict[str, Any] | None:
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


def run_tui_app(parser, args) -> None:
    """Entrypoint for the native Python TUI experience."""
    import os

    from slavv_python.engine import SlavvPipeline
    from slavv_python.storage.loaders import load_tiff_volume

    config = run_wizard_if_supported()
    if not config or not config.get("execute_now"):
        return

    input_path = config.get("input_path") or "volume.tif"
    output_dir = config.get("output_dir") or "./slavv_output"
    profile = config.get("profile") or "paper"

    try:
        image = load_tiff_volume(input_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    pipeline = SlavvPipeline()
    params = {"profile": profile}

    print("\nStarting pipeline execution...")
    
    # Run the pipeline without the Textual dashboard for now as it causes 
    # threading complexity with standard output, or use the Textual monitor if desired.
    # Currently, we'll just run it directly. If the textual monitor is needed, 
    # it requires a threaded approach which can be built out using Textual's worker API.
    pipeline.run(
        image,
        params,
        run_dir=os.path.join(output_dir, "_slavv_run"),
    )

    print("\nPipeline run completed successfully.")
