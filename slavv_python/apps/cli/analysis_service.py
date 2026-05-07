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
    import numpy as np

    v_data = results.get("vertices", {})
    e_data = results.get("edges", {})
    n_data = results.get("network", {})
    p_data = results.get("parameters", {})

    strands = n_data.get("strands", [])
    bifurcations = n_data.get("bifurcations", np.array([]))
    positions = v_data.get("positions", np.array([]))
    radii = v_data.get("radii_microns", np.array([]))
    microns_per_voxel = p_data.get("microns_per_voxel", [1.0, 1.0, 1.0])
    image_shape = results.get("image_shape", (1, 1, 1))
    edge_energies = e_data.get("energies", None)

    stats_dict = statistics_fn(
        strands,
        bifurcations,
        positions,
        radii,
        microns_per_voxel,
        image_shape,
        edge_energies=edge_energies,
    )
    return list(stats_dict.items())


__all__ = ["calculate_exported_network_stats"]
