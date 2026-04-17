"""Dashboard-specific plotting helpers for SLAVV network plots."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go


def add_summary_dashboard_traces(
    fig: go.Figure,
    vertices: dict[str, Any],
    network: dict[str, Any],
    parameters: dict[str, Any],
) -> None:
    """Populate the summary dashboard subplots with derived traces."""
    vertex_positions = vertices["positions"]
    microns_per_voxel = parameters.get("microns_per_voxel", [1.0, 1.0, 1.0])

    if len(vertex_positions) > 0:
        x_coords = vertex_positions[:, 1] * microns_per_voxel[1]
        y_coords = vertex_positions[:, 0] * microns_per_voxel[0]
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker={"size": 4, "color": "red"},
                name="Vertices",
            ),
            row=1,
            col=1,
        )

    strand_lengths = []
    for strand in network["strands"]:
        if len(strand) > 1:
            length = 0.0
            for i in range(len(strand) - 1):
                pos1 = vertex_positions[strand[i]] * microns_per_voxel
                pos2 = vertex_positions[strand[i + 1]] * microns_per_voxel
                length += float(np.linalg.norm(pos2 - pos1))
            strand_lengths.append(length)

    if strand_lengths:
        fig.add_trace(go.Histogram(x=strand_lengths, nbinsx=15, name="Strand Lengths"), row=1, col=2)

    radii = vertices.get("radii_microns", vertices.get("radii", []))
    if len(radii) > 0:
        fig.add_trace(go.Histogram(x=radii, nbinsx=15, name="Radii"), row=2, col=1)

    if len(vertex_positions) > 0:
        depths = vertex_positions[:, 2] * microns_per_voxel[2]
        depth_counts, depth_bins = np.histogram(depths, bins=10)
        bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
        fig.add_trace(go.Bar(x=bin_centers, y=depth_counts, name="Vertex Count by Depth"), row=2, col=2)
