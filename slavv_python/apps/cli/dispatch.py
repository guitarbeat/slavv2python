"""Dispatch helpers for the SLAVV CLI entrypoint."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from .commands import (
    _handle_analyze_command,
    _handle_info_command,
    _handle_plot_command,
    _handle_run_command,
    _handle_status_command,
)

CLI_COMMAND_HANDLERS: dict[str | None, Callable[..., None]] = {
    "run": _handle_run_command,
    "analyze": _handle_analyze_command,
    "plot": _handle_plot_command,
    "info": _handle_info_command,
    "status": _handle_status_command,
}


def dispatch_cli_command(parser, args) -> None:
    """Execute the handler for the requested CLI command."""
    if args.tui:
        # Launch interactive TUI wizard/monitor
        from .tui import run_tui_app

        run_tui_app(parser, args)
        return

    handler = CLI_COMMAND_HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(0 if args.command is None else 1)
    handler(args)


__all__ = ["CLI_COMMAND_HANDLERS", "dispatch_cli_command"]
