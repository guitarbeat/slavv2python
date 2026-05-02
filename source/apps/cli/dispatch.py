"""Grouped CLI dispatch surface for SLAVV."""

from __future__ import annotations

from ..cli_dispatch import CLI_COMMAND_HANDLERS, dispatch_cli_command

__all__ = [
    "CLI_COMMAND_HANDLERS",
    "dispatch_cli_command",
]
