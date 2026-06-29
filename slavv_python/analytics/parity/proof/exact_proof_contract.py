"""Exact-route proof stage contract shared by loaders and comparators."""

from __future__ import annotations

EXACT_STAGE_ORDER: tuple[str, ...] = ("energy", "vertices", "edges", "network")
EXACT_STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "energy": ("energy", "scale_indices", "energy_4d", "lumen_radius_microns"),
    "vertices": ("positions", "scales", "energies"),
    "edges": (
        "connections",
        "traces",
        "scale_traces",
        "energy_traces",
        "energies",
        "bridge_vertex_positions",
        "bridge_vertex_scales",
        "bridge_vertex_energies",
        "bridge_edges",
    ),
    "network": (
        "strands",
        "bifurcations",
        "strand_subscripts",
        "strand_energy_traces",
        "mean_strand_energies",
        "vessel_directions",
    ),
}
BRIDGE_EDGE_FIELDS: tuple[str, ...] = (
    "connections",
    "traces",
    "scale_traces",
    "energy_traces",
    "energies",
)
