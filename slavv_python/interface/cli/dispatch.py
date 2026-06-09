"""Dispatch helpers for the SLAVV CLI entrypoint."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from .commands import (
    _handle_analyze_command,
    _handle_info_command,
    _handle_monitor_command,
    _handle_plot_command,
    _handle_run_command,
    _handle_status_command,
)


def _handle_jobs_command(args) -> None:
    """Delegate to jobs subcommand handler."""
    from .jobs import main as jobs_main

    # Re-parse with jobs-specific parser
    jobs_main(sys.argv[2:])  # Skip 'slavv jobs'


CLI_COMMAND_HANDLERS: dict[str | None, Callable[..., None]] = {
    "run": _handle_run_command,
    "analyze": _handle_analyze_command,
    "plot": _handle_plot_command,
    "info": _handle_info_command,
    "status": _handle_status_command,
    "monitor": _handle_monitor_command,
    "jobs": _handle_jobs_command,
}


def dispatch_cli_command(parser, args) -> None:
    """Execute the handler for the requested CLI command."""
    handler = CLI_COMMAND_HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(0 if args.command is None else 1)
    handler(args)


__all__ = ["CLI_COMMAND_HANDLERS", "dispatch_cli_command"]
