"""Preferred internal name for vertex candidate detection helpers."""

from __future__ import annotations

from .._vertices import candidates as _legacy_candidates

__all__ = [name for name in dir(_legacy_candidates) if not name.startswith("__")]


def __getattr__(name: str):
    return getattr(_legacy_candidates, name)


def __dir__() -> list[str]:
    return sorted(__all__)
