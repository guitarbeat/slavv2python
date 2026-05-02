"""Compatibility wrapper for flat 2D network-plot helpers."""

from __future__ import annotations

from source.visualization._compat import bind_legacy_module

_LEGACY_MODULE, __all__ = bind_legacy_module("source.visualization.network_plots.spatial_2d")


def __getattr__(name: str):
    return getattr(_LEGACY_MODULE, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
