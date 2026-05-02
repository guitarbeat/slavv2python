"""Preferred internal name for run-tracking reset helpers."""

from __future__ import annotations

from .._run_state import reset as _legacy_reset

__all__ = [name for name in dir(_legacy_reset) if not name.startswith("__")]


def __getattr__(name: str):
    return getattr(_legacy_reset, name)


def __dir__() -> list[str]:
    return sorted(__all__)
