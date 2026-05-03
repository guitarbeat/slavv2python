"""Information helpers for the SLAVV CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

from .cli_reporting import build_info_lines


def load_info_lines(
        *,
        version: str,
        system_info: Mapping[str, object],
) -> list[str]:
    """Build the printable lines for the CLI info command."""
    return build_info_lines(version, system_info)


__all__ = ["load_info_lines"]
