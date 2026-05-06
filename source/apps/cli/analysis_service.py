"""Analysis helpers for exported-network CLI commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def calculate_exported_network_stats(
    results: Mapping[str, Any],
    *,
    statistics_fn,
) -> list[tuple[str, object]]:
    """Compute and format aggregate statistics from an exported network dict."""
    return statistics_fn(
        results["vertices"]["positions"],
        results["vertices"]["radii_microns"],
        results["parameters"].get("microns_per_voxel", [1.0, 1.0, 1.0]),
        results["image_shape"],
    )


__all__ = ["calculate_exported_network_stats"]
