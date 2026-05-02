"""Preferred internal name for edge cleanup helpers."""

from __future__ import annotations

from .._edge_selection import cleanup as _legacy_cleanup

__all__ = [name for name in dir(_legacy_cleanup) if not name.startswith("__")]


def __getattr__(name: str):
    return getattr(_legacy_cleanup, name)


def __dir__() -> list[str]:
    return sorted(__all__)
