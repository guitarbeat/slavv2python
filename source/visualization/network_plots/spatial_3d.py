"""3D spatial plotting helpers for SLAVV network visualizations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ...utils import calculate_path_length
from .helpers import add_colorbar
from .layout import plot_3d_layout

logger = logging.getLogger(__name__)


def plot_3d_network(
    self,
    vertices: dict[str, Any],
    edges: dict[str, Any],
    network: dict[str, Any],
    parameters: dict[str, Any],
    color_by: str = "energy",
    show_vertices: bool = True,
    show_edges: bool = True,
    show_bifurcations: bool = True,
    opacity_by: str | None = None,
) -> go.Figure:
    """Create 3D network visualization.

    Optimized version using merged traces for performance.

    Parameters
    ----------
    vertices : Dict[str, Any]
        Vertex data.
    edges : Dict[str, Any]
        Edge data.
    network : Dict[str, Any]
        Network data.
    parameters : Dict[str, Any]
        Processing parameters.
    color_by : str, optional
        Coloring scheme ("energy", "depth", "strand_id", "radius", "length"), by default "energy".
    show_vertices : bool, optional
        Whether to show vertices, by default True.
    show_edges : bool, optional
        Whether to show edges, by default True.
    show_bifurcations : bool, optional
        Whether to highlight bifurcations, by default True.
    opacity_by : Optional[str], optional
        Attribute controlling edge opacity (currently supports "depth"), by default ``None``.

    Returns
    -------
    go.Figure
        3D Plotly figure of the vascular network.
    """
    logger.info(
        f"Creating 3D network plot with {color_by} coloring"
        + (f" and {opacity_by} opacity" if opacity_by else "")
    )

    fig = go.Figure()

    vertex_positions = vertices["positions"]
    vertex_energies = vertices["energies"]
    vertex_radii = vertices.get("radii_microns", vertices.get("radii", []))
    edge_traces = edges["traces"]
    bifurcations = network.get("bifurcations", [])

    # Convert to physical units
    microns_per_voxel = parameters.get("microns_per_voxel", [1.0, 1.0, 1.0])

    # Plot edges as 3D lines
    if show_edges and edge_traces:
        valid_traces_indices = [i for i, t in enumerate(edge_traces) if len(t) >= 2]

        # Pre-calculate strand IDs if needed
        strand_ids_map = {}
        if color_by == "strand_id":
            connections = edges.get("connections", [])
            pair_to_index = {
                tuple(sorted(map(int, conn))): idx for idx, conn in enumerate(connections)
            }
            for sid, strand in enumerate(network.get("strands", [])):
                for v0, v1 in zip(strand[:-1], strand[1:]):
                    idx = pair_to_index.get(tuple(sorted((int(v0), int(v1)))))
                    if idx is not None:
                        strand_ids_map[idx] = sid

        # Arrays to hold merged data
        x_all = []
        y_all = []
        z_all = []
        color_values = []
        custom_data = []  # [edge_index, length]

        # Collect values for colorbar later
        edge_values_for_cbar = []

        # Prepare value for each edge first
        edge_val_map = {}

        # Determine values for coloring
        for i in valid_traces_indices:
            trace = np.array(edge_traces[i])

            # Value calculation
            val = 0.0
            if color_by == "depth":
                val = np.mean(trace[:, 2] * microns_per_voxel[2])
            elif color_by == "energy":
                energies = edges.get("energies", [])
                val = energies[i] if i < len(energies) else 0.0
            elif color_by == "length":
                val = calculate_path_length(trace * microns_per_voxel)
            elif color_by == "radius":
                connections = edges.get("connections", [])
                if i < len(connections):
                    v0, v1 = connections[i]
                    r0 = vertex_radii[int(v0)] if int(v0) >= 0 else 0
                    r1 = (
                        vertex_radii[int(v1)]
                        if int(v1) >= 0 and int(v1) < len(vertex_radii)
                        else r0
                    )
                    val = (r0 + r1) / 2.0
            elif color_by == "strand_id":
                val = strand_ids_map.get(i, -1)

            edge_val_map[i] = val
            edge_values_for_cbar.append(val)

        # Note: opacity_by='depth' is disabled in optimized merged trace mode
        # as per-segment opacity is not supported efficiently in a single go.Scatter3d trace.

        # Loop to build arrays
        for idx in valid_traces_indices:
            trace = np.array(edge_traces[idx])
            x = trace[:, 1] * microns_per_voxel[1]
            y = trace[:, 0] * microns_per_voxel[0]
            z = trace[:, 2] * microns_per_voxel[2]

            x_all.extend(x)
            y_all.extend(y)
            z_all.extend(z)

            x_all.append(None)
            y_all.append(None)
            z_all.append(None)

            val = edge_val_map[idx]
            length = calculate_path_length(trace * microns_per_voxel)

            # Repeat value for all points + None
            color_values.extend([val] * (len(x) + 1))

            # Custom data
            # We can store [edge_index, length]
            # Repeat for all points + None
            cd = [[idx, length]] * (len(x) + 1)
            custom_data.extend(cd)

        # Colorscale selection
        colorscale = self.color_schemes.get(color_by, "Viridis")
        if color_by == "strand_id":
            colorscale = "Turbo"

        fig.add_trace(
            go.Scatter3d(
                x=x_all,
                y=y_all,
                z=z_all,
                mode="lines",
                line={"color": color_values, "colorscale": colorscale, "width": 4},
                name="Edges",
                customdata=custom_data,
                hovertemplate="Edge %{customdata[0]}<br>Length: %{customdata[1]:.1f} μm<extra></extra>",
                opacity=1.0,  # Uniform opacity as merged trace doesn't support per-segment opacity easily
            )
        )

        # Add colorbar
        if color_by in {"depth", "energy", "radius", "length"}:
            add_colorbar(
                fig, np.array(edge_values_for_cbar), colorscale, color_by.title(), is_3d=True
            )

    # Plot vertices
    if show_vertices and len(vertex_positions) > 0:
        x_coords = vertex_positions[:, 1] * microns_per_voxel[1]  # X
        y_coords = vertex_positions[:, 0] * microns_per_voxel[0]  # Y
        z_coords = vertex_positions[:, 2] * microns_per_voxel[2]  # Z

        # Color vertices
        if color_by == "depth":
            colors = z_coords
            colorscale = "Viridis"
        elif color_by == "energy":
            colors = vertex_energies
            colorscale = "RdBu_r"
        elif color_by == "radius":
            colors = vertex_radii
            colorscale = "Plasma"
        else:
            colors = "red"
            colorscale = None

        edge_colorbar = (
            show_edges and edge_traces and color_by in {"depth", "energy", "radius", "length"}
        )
        marker_dict = {
            "size": 6,
            "color": colors,
            "colorscale": colorscale,
            "showscale": bool(colorscale and not edge_colorbar),
            "colorbar": ({"title": color_by.title()} if colorscale and not edge_colorbar else None),
            "line": {"width": 1, "color": "black"},
        }

        fig.add_trace(
            go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode="markers",
                marker=marker_dict,
                name="Vertices",
                hovertemplate="Vertex<br>Energy: %{customdata[0]:.3f}<br>Radius: %{customdata[1]:.2f} μm<extra></extra>",
                customdata=np.column_stack([vertex_energies, vertex_radii]),
            )
        )

    # Highlight bifurcations
    if show_bifurcations and len(bifurcations) > 0:
        bif_positions = vertex_positions[bifurcations]
        x_coords = bif_positions[:, 1] * microns_per_voxel[1]
        y_coords = bif_positions[:, 0] * microns_per_voxel[0]
        z_coords = bif_positions[:, 2] * microns_per_voxel[2]

        fig.add_trace(
            go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode="markers",
                marker={
                    "size": 10,
                    "color": "yellow",
                    "symbol": "diamond",
                    "line": {"width": 2, "color": "black"},
                },
                name="Bifurcations",
                hovertemplate="Bifurcation<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title="3D Vascular Network",
        scene={
            "xaxis_title": "X (μm)",
            "yaxis_title": "Y (μm)",
            "zaxis_title": "Z (μm)",
            "aspectmode": "data",
        },
        showlegend=True,
        width=800,
        height=600,
    )

    fig.update_layout(**plot_3d_layout("3D Vascular Network", showlegend=True))
    return fig


def animate_strands_3d(
    self,
    vertices: dict[str, Any],
    edges: dict[str, Any],
    network: dict[str, Any],
    parameters: dict[str, Any],
) -> go.Figure:
    """Animate strands sequentially in 3D.

    Parameters
    ----------
    vertices, edges, network, parameters : dict
        Standard SLAVV network outputs and processing parameters. The
        animation iterates over ``network['strands']`` and renders each
        strand's edges in turn.

    Returns
    -------
    go.Figure
        Plotly figure with animation frames for each strand.
    """
    logger.info("Creating 3D strand animation")

    microns_per_voxel = parameters.get("microns_per_voxel", [1.0, 1.0, 1.0])
    vertex_positions = vertices.get("positions", np.empty((0, 3)))

    # Base figure with all vertices shown as reference
    x = vertex_positions[:, 1] * microns_per_voxel[1]
    y = vertex_positions[:, 0] * microns_per_voxel[0]
    z = vertex_positions[:, 2] * microns_per_voxel[2]
    vertex_scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker={"size": 4, "color": "lightgray"},
        name="Vertices",
    )

    connections = edges.get("connections", [])
    traces = edges.get("traces", [])
    pair_to_index = {tuple(sorted(map(int, conn))): idx for idx, conn in enumerate(connections)}

    colors = px.colors.qualitative.Set3
    frames: list[go.Frame] = []
    for sid, strand in enumerate(network.get("strands", [])):
        edge_traces = []
        color = colors[sid % len(colors)]
        for v0, v1 in zip(strand[:-1], strand[1:]):
            idx = pair_to_index.get(tuple(sorted((int(v0), int(v1)))))
            if idx is None:
                continue
            trace = np.asarray(traces[idx])
            edge_traces.append(
                go.Scatter3d(
                    x=trace[:, 1] * microns_per_voxel[1],
                    y=trace[:, 0] * microns_per_voxel[0],
                    z=trace[:, 2] * microns_per_voxel[2],
                    mode="lines",
                    line={"color": color, "width": 4},
                    name=f"Strand {sid}",
                )
            )
        frames.append(go.Frame(data=[vertex_scatter, *edge_traces], name=str(sid)))

    fig = go.Figure(
        data=frames[0].data if frames else [vertex_scatter],
        frames=frames,
    )

    steps = [
        {
            "label": str(i),
            "method": "animate",
            "args": [
                [str(i)],
                {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"},
            ],
        }
        for i in range(len(frames))
    ]

    fig.update_layout(
        **plot_3d_layout(
            "Animated 3D Strands",
            showlegend=False,
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [{"label": "Play", "method": "animate", "args": [None]}],
                }
            ],
            sliders=[{"active": 0, "steps": steps, "currentvalue": {"prefix": "Strand: "}}],
        )
    )

    return fig


def plot_flow_field(
    self,
    edges: dict[str, Any],
    parameters: dict[str, Any],
) -> go.Figure:
    """Render edge directions as a 3D flow field.

    Parameters
    ----------
    edges : dict
        Edge data containing ``traces`` where each trace is an array of
        voxel coordinates in ``(y, x, z)`` order.
    parameters : dict
        Processing parameters with ``microns_per_voxel`` for unit
        conversion.

    Returns
    -------
    go.Figure
        Plotly figure with cones indicating edge orientations.
    """
    logger.info("Creating flow field visualization")

    traces = edges.get("traces", [])
    if not traces:
        return go.Figure()

    microns_per_voxel = parameters.get("microns_per_voxel", [1.0, 1.0, 1.0])

    x, y, z, u, v, w = [], [], [], [], [], []
    for trace in traces:
        arr = np.asarray(trace)
        if len(arr) < 2:
            continue
        start = arr[0]
        end = arr[-1]
        mid = (start + end) / 2.0
        direction = end - start
        y.append(mid[0] * microns_per_voxel[0])
        x.append(mid[1] * microns_per_voxel[1])
        z.append(mid[2] * microns_per_voxel[2])
        v.append(direction[0] * microns_per_voxel[0])
        u.append(direction[1] * microns_per_voxel[1])
        w.append(direction[2] * microns_per_voxel[2])

    cone = go.Cone(x=x, y=y, z=z, u=u, v=v, w=w, colorscale="Blues", showscale=False)
    fig = go.Figure(data=[cone])
    fig.update_layout(**plot_3d_layout("Edge Flow Field", showlegend=False))
    return fig
