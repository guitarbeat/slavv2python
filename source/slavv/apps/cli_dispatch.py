"""Dispatch helpers for the SLAVV CLI entrypoint."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from .cli_commands import (
    _handle_analyze_command,
    _handle_import_matlab_command,
    _handle_info_command,
    _handle_parity_proof_command,
    _handle_plot_command,
    _handle_run_command,
    _handle_status_command,
)

CLI_COMMAND_HANDLERS: dict[str, Callable[[object], None]] = {
    "analyze": _handle_analyze_command,
    "import-matlab": _handle_import_matlab_command,
    "info": lambda _args: _handle_info_command(),
    "parity-proof": _handle_parity_proof_command,
    "plot": _handle_plot_command,
    "run": _handle_run_command,
    "status": _handle_status_command,
}


def dispatch_cli_command(parser, args) -> None:
    """Dispatch parsed CLI args to the matching command handler."""
    handler = CLI_COMMAND_HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(0 if args.command is None else 1)
    handler(args)


__all__ = ["CLI_COMMAND_HANDLERS", "dispatch_cli_command"]
