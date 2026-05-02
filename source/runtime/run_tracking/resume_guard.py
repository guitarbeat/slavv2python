"""Preferred internal name for run-tracking resume guards."""

from __future__ import annotations

from .._run_state import resume_guard as _legacy_resume_guard

__all__ = [name for name in dir(_legacy_resume_guard) if not name.startswith("__")]


def __getattr__(name: str):
    return getattr(_legacy_resume_guard, name)


def __dir__() -> list[str]:
    return sorted(__all__)
