"""Compatibility shim for legacy native Hessian response."""

from __future__ import annotations

from ..energy_internal import hessian_response as _hessian

__all__ = [name for name in dir(_hessian) if not name.startswith("__")]


def __getattr__(name: str):
    return getattr(_hessian, name)


def __dir__() -> list[str]:
    return sorted(__all__)
