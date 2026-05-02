"""Preferred internal name for run-tracking lifecycle helpers."""

from __future__ import annotations

from .._run_state import lifecycle as _legacy_lifecycle

__all__ = [name for name in dir(_legacy_lifecycle) if not name.startswith("__")]


def __getattr__(name: str):
    return getattr(_legacy_lifecycle, name)


def __dir__() -> list[str]:
    return sorted(__all__)
