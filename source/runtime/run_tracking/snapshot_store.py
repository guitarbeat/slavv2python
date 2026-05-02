"""Preferred internal name for run-tracking snapshot persistence."""

from __future__ import annotations

from .._run_state import snapshot_store as _legacy_snapshot_store

__all__ = [name for name in dir(_legacy_snapshot_store) if not name.startswith("__")]


def __getattr__(name: str):
    return getattr(_legacy_snapshot_store, name)


def __dir__() -> list[str]:
    return sorted(__all__)
