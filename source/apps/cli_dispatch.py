"""Dispatch helpers for the SLAVV CLI entrypoint."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from .cli_commands import (
    _handle_analyze_command,
    _handle_info_command,
    _handle_plot_command,
    _handle_run_command,
    _handle_status_command,
)

CLI_COMMAND_HANDLERS: dict[str, Callable[[object], None]] = {
    "analyze": _handle_analyze_command,
    "info": lambda _args: _handle_info_command(),
    "plot": _handle_plot_command,
    "run": _handle_run_command,
    "status": _handle_status_command,
}


def dispatch_cli_command(parser, args) -> None:
    """Dispatch parsed CLI args to the matching command handler."""
    if getattr(args, "tui", False) or args.command is None:
        from .cli.tui import is_tui_available, run_monitor_if_supported, run_wizard_if_supported

        if is_tui_available():
            result = run_wizard_if_supported()
            if result and result.get("execute_now"):
                print("\n⚙️ Settings configured! Launching Live Monitor...")
                run_monitor_if_supported()
            return
        if getattr(args, "tui", False):
            print("Error: TUI dependencies (questionary, textual) are not installed.")
            print('Please run: pip install -e ".[tui]"')
            sys.exit(1)

    handler = CLI_COMMAND_HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(0 if args.command is None else 1)
    handler(args)


__all__ = ["CLI_COMMAND_HANDLERS", "dispatch_cli_command"]
