"""Compatibility wrapper for flat pipeline runner helpers."""

from __future__ import annotations

from source.workflows._compat import bind_legacy_module

_LEGACY_MODULE, __all__ = bind_legacy_module("source.workflows.pipeline.execution")


def __getattr__(name: str):
    return getattr(_LEGACY_MODULE, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
