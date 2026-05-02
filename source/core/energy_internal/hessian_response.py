"""Preferred internal name for the native Hessian response backend."""

from __future__ import annotations

from .._energy import native_hessian as _legacy_native_hessian

__all__ = [name for name in dir(_legacy_native_hessian) if not name.startswith("__")]


def __getattr__(name: str):
    return getattr(_legacy_native_hessian, name)


def __dir__() -> list[str]:
    return sorted(__all__)
