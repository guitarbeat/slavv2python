"""Preferred internal name for run-tracking progress helpers."""

from __future__ import annotations

from .._run_state import progress as _legacy_progress

__all__ = [name for name in dir(_legacy_progress) if not name.startswith("__")]


def __getattr__(name: str):
    return getattr(_legacy_progress, name)


def __dir__() -> list[str]:
    return sorted(__all__)
