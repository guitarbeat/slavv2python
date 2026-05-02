"""Preferred internal name for SLAVV energy backends."""

from __future__ import annotations

from .._energy import backends as _legacy_backends

__all__ = [name for name in dir(_legacy_backends) if not name.startswith("__")]


def __getattr__(name: str):
    return getattr(_legacy_backends, name)


def __dir__() -> list[str]:
    return sorted(__all__)
