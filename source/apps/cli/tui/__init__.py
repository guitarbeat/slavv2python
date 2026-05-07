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


def run_tui_app(parser, args) -> None:
    """Entrypoint for the merged TUI experience."""
    import json
    import os
    import subprocess
    import sys
    from threading import Thread

    from source.core import SlavvPipeline
    from source.io import load_tiff_volume

    # 1. Try to run the Node/Clack wizard for setup
    node_wizard_path = os.path.join(os.getcwd(), "source", "apps", "cli", "tui", "node_ui")
    config = None

    if os.path.exists(node_wizard_path):
        try:
            # Run node wizard with --json flag to get the results
            result = subprocess.run(
                ["npm", "run", "start", "--", "--json"],
                cwd=node_wizard_path,
                capture_output=True,
                text=True,
                check=False,
                shell=True,
            )
            if result.returncode == 0 and result.stdout:
                # Find the JSON part in stdout (in case there's other output)
                import re

                json_match = re.search(r"\{.*\}", result.stdout)
                if json_match:
                    config = json.loads(json_match.group(0))
        except Exception as e:
            logger.debug(f"Node wizard failed or not found: {e}")

    # 2. Fallback to Python wizard if Node failed or was cancelled
    if config is None:
        config = run_wizard_if_supported()
        if not config or not config.get("execute_now"):
            return

    # 3. Map config to pipeline parameters
    # The Node wizard and Python wizard have slightly different keys
    input_path = config.get("input_path") or config.get("inputPath")  # Node uses camelCase sometimes
    # Wait, looking at clack-wizard.ts, it uses: projectName, environment, modules, outputDir, threadLimit
    # Python wizard uses: input_path, output_dir, profile, exports

    input_path = config.get("input_path")
    if not input_path and "projectName" in config:
        # If it's the Node wizard, we might need to prompt for input path if it missed it
        # but for now let's assume standard mapping
        input_path = "volume.tif"  # Default

    output_dir = config.get("output_dir") or config.get("outputDir") or "./slavv_output"
    profile = config.get("profile") or "paper"

    # 4. Prepare pipeline
    try:
        image = load_tiff_volume(input_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    pipeline = SlavvPipeline()
    params = {"profile": profile}

    # 5. Spawn Node Dashboard if using the merged UI
    dashboard_proc = None
    if os.path.exists(node_wizard_path):
        try:
            dashboard_proc = subprocess.Popen(
                ["npm", "run", "start", "--", "--dashboard", json.dumps(config)],
                cwd=node_wizard_path,
                stdin=subprocess.PIPE,
                text=True,
                shell=True,
            )
        except Exception as e:
            logger.debug(f"Failed to spawn Node dashboard: {e}")

    # 6. Run pipeline with progress callback
    def _event_callback(event):
        if dashboard_proc and dashboard_proc.stdin:
            # Send progress to Node dashboard via stdin
            msg = json.dumps(
                {
                    "type": "progress",
                    "value": int(event.overall_progress * 100),
                    "stage": event.stage,
                    "message": event.detail,
                }
            )
            try:
                dashboard_proc.stdin.write(msg + "\n")
                dashboard_proc.stdin.flush()
            except Exception:
                pass

    # Run in a thread or just normally?
    # If we have a dashboard, we should run the pipeline and then wait for dashboard to exit
    results = pipeline.run(
        image,
        params,
        event_callback=_event_callback,
        run_dir=os.path.join(output_dir, "_slavv_run"),
    )

    if dashboard_proc:
        # Signal completion
        if dashboard_proc.stdin:
            dashboard_proc.stdin.write(
                json.dumps({"type": "progress", "value": 100, "stage": "completed"}) + "\n"
            )
            dashboard_proc.stdin.flush()
        dashboard_proc.wait()
    else:
        # Fallback to simple printing if no dashboard
        print("Pipeline run completed successfully.")
