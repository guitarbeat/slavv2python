"""Preferred grouped Streamlit dashboard service surface."""

from __future__ import annotations

from .._compat import bind_legacy_module

_LEGACY_MODULE, __all__ = bind_legacy_module("source.apps.services.dashboard")


def __getattr__(name: str):
    return getattr(_LEGACY_MODULE, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
