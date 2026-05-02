"""Compatibility wrapper for flat network-plot layout helpers."""

from __future__ import annotations

from source.visualization._compat import bind_legacy_module

_LEGACY_MODULE, __all__ = bind_legacy_module("source.visualization.network_plots.layout")


def __getattr__(name: str):
    return getattr(_LEGACY_MODULE, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
