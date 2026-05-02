"""Compatibility wrapper for legacy underscore energy-gradient helpers."""

from __future__ import annotations

from source.core._compat import bind_legacy_module

_LEGACY_MODULE, __all__ = bind_legacy_module("source.core.energy_internal.energy_gradients")


def __getattr__(name: str):
    return getattr(_LEGACY_MODULE, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
