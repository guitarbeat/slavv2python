"""
Visualization module for SLAVV results

This module provides comprehensive visualization capabilities for vascular networks
including 2D/3D plotting, statistical analysis, and export functionality.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils import calculate_path_length
from .network_plot_dashboard import add_summary_dashboard_traces
from .network_plot_exports import export_casx, export_csv, export_json, export_mat, export_vmv
from .network_plot_helpers import NETWORK_COLOR_SCHEMES, add_colorbar, map_values_to_colors
from .network_plot_layout import (
    axis_labels,
    distribution_layout,
    empty_figure,
    plot_2d_layout,
    plot_3d_layout,
    plot_slice_layout,
    select_plot_axes,
    summary_dashboard_layout,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkVisualizer:
    """
    Comprehensive visualization class for SLAVV results
    """

    def __init__(self):
        self.color_schemes = dict(NETWORK_COLOR_SCHEMES)

    def plot_2d_network(
        self,
        vertices: dict[str, Any],
        edges: dict[str, Any],
        network: dict[str, Any],
        parameters: dict[str, Any],
        color_by: str = "energy",
        projection_axis: int = 2,
        show_vertices: bool = True,
        show_edges: bool = True,
        show_bifurcations: bool = True,
    ) -> go.Figure:
        """
        Create 2D network visualization with projection

        Args:
            vertices: Vertex data
            edges: Edge data
            network: Network data
            parameters: Processing parameters
            color_by: Coloring scheme ('energy', 'depth', 'strand_id', 'radius', 'length')
            projection_axis: Axis to project along (0=Y, 1=X, 2=Z)
            show_vertices: Whether to show vertices
            show_edges: Whether to show edges
            show_bifurcations: Whether to highlight bifurcations
        """
        logger.info(f"Creating 2D network plot with {color_by} coloring")

        fig = go.Figure()

        vertex_positions = vertices["positions"]
        vertex_energies = vertices["energies"]
        vertex_radii = vertices.get("radii_microns", vertices.get("radii", []))
        edge_traces = edges["traces"]
        bifurcations = network.get("bifurcations", [])

        x_axis, y_axis = select_plot_axes(projection_axis)
        x_label, y_label = axis_labels(x_axis, y_axis)

        # Convert to physical units
        microns_per_voxel = parameters.get("microns_per_voxel", [1.0, 1.0, 1.0])

        # Plot edges
        if show_edges and edge_traces:
            valid_traces = [np.array(t) for t in edge_traces if len(t) >= 2]
            # Determine values and colors for all edges
            edge_colors: list[str] = []
            strand_ids: list[int] = []
            values: np.ndarray | None = None

            # 1. Calculate values based on color_by
            if color_by == "depth":
                depths = [
                    np.mean(t[:, projection_axis]) * microns_per_voxel[projection_axis]
                    for t in valid_traces
                ]
                values = np.array(depths)
            elif color_by == "energy":
                energies = edges.get("energies", [])
                if len(energies) == len(valid_traces):
                    values = np.asarray(energies)
                else:
                    edge_colors = ["blue"] * len(valid_traces)
            elif color_by == "length":
                lengths = [
                    calculate_path_length(trace * microns_per_voxel) for trace in valid_traces
                ]
                values = np.asarray(lengths)

            elif color_by == "radius":
                connections = edges.get("connections", [])
                if len(connections) == len(valid_traces) and len(vertex_radii) > 0:
                    radii = []
                    for v0, v1 in connections:
                        r0 = vertex_radii[int(v0)] if int(v0) >= 0 else 0
                        r1 = (
                            vertex_radii[int(v1)]
                            if int(v1) >= 0 and int(v1) < len(vertex_radii)
                            else r0
                        )
                        radii.append((r0 + r1) / 2.0)
                    values = np.asarray(radii)
                else:
                    edge_colors = ["blue"] * len(valid_traces)
            # Quantize values to reduce number of unique colors (and thus traces)
            if values is not None:
                # Use 64 bins for high fidelity but reasonable performance
                n_bins = 64
                vmin, vmax = np.min(values), np.max(values)
                if vmax > vmin:
                    # Quantize to bins
                    bins = np.linspace(vmin, vmax, n_bins)
                    # Use searchsorted instead of digitize for better performance with float arrays
                    # indices will be 0..n_bins
                    indices = np.searchsorted(bins, values)
                    # Clip to valid range (searchsorted can return len(bins))
                    indices = np.clip(indices, 0, len(bins) - 1)
                    quantized_values = bins[indices]
                    edge_colors = map_values_to_colors(
                        quantized_values, self.color_schemes[color_by]
                    )
                else:
                    edge_colors = map_values_to_colors(values, self.color_schemes[color_by])
            elif color_by == "strand_id":
                connections = edges.get("connections", [])
                pair_to_index = {
                    tuple(sorted(map(int, conn))): idx for idx, conn in enumerate(connections)
                }
                strand_ids = [-1] * len(valid_traces)
                for sid, strand in enumerate(network.get("strands", [])):
                    for v0, v1 in zip(strand[:-1], strand[1:]):
                        idx = pair_to_index.get(tuple(sorted((int(v0), int(v1)))))
                        if idx is not None:
                            strand_ids[idx] = sid
                colors = px.colors.qualitative.Set3
                edge_colors = [
                    colors[sid % len(colors)] if sid >= 0 else "blue" for sid in strand_ids
                ]
            else:
                edge_colors = ["blue"] * len(valid_traces)

            # 2. Group edges by color to reduce trace count (using scattergl for performance)
            batched_traces: dict[str, dict[str, list]] = {}

            for i, trace in enumerate(valid_traces):
                color = edge_colors[i]
                if color not in batched_traces:
                    batched_traces[color] = {"x": [], "y": [], "customdata": [], "names": []}

                # Get coordinates
                xs = trace[:, x_axis] * microns_per_voxel[x_axis]
                ys = trace[:, y_axis] * microns_per_voxel[y_axis]

                # Append to lists with None separator
                batched_traces[color]["x"].extend(xs)
                batched_traces[color]["x"].append(None)
                batched_traces[color]["y"].extend(ys)
                batched_traces[color]["y"].append(None)

                # Metadata for hover
                length = calculate_path_length(trace * microns_per_voxel)
                if color_by == "strand_id":
                    sid = strand_ids[i]
                    name = f"Strand {sid}"
                    # We can't really set 'name' per point, but we can put it in customdata
                    # For strand_id, we might want to group strictly by strand ID instead of color if we want precise legend toggle
                    # But for performance with many strands, grouping by color is safer.
                else:
                    name = f"Edge {i}"

                # Create customdata for each point + None
                # Format: [Edge Index, Length, Name]
                edge_meta = [i, length, name]
                batched_traces[color]["customdata"].extend([edge_meta] * len(xs))
                batched_traces[color]["customdata"].append([None, None, None])

            showlegend = False

            # 3. Create merged traces
            for color, data in batched_traces.items():
                # Determine legend name
                name = "Edges"
                fig.add_trace(
                    go.Scattergl(
                        x=data["x"],
                        y=data["y"],
                        mode="lines",
                        line={"color": color, "width": 2},
                        name=name,
                        showlegend=showlegend,
                        customdata=data["customdata"],
                        hovertemplate=(
                            "%{customdata[2]}<br>Length: %{customdata[1]:.1f} μm<extra></extra>"
                        ),
                    )
                )

            # Add colorbar if applicable
            if color_by in {"depth", "energy", "radius", "length"} and values is not None:
                add_colorbar(
                    fig, values, self.color_schemes[color_by], color_by.title(), is_3d=False
                )

        # Plot vertices
        if show_vertices and len(vertex_positions) > 0:
            x_coords = vertex_positions[:, x_axis] * microns_per_voxel[x_axis]
            y_coords = vertex_positions[:, y_axis] * microns_per_voxel[y_axis]

            # Color vertices
            if color_by == "depth":
                colors = vertex_positions[:, projection_axis] * microns_per_voxel[projection_axis]
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
                "size": 8,
                "color": colors,
                "colorscale": colorscale,
                "showscale": bool(colorscale and not edge_colorbar),
                "colorbar": (
                    {"title": color_by.title()} if colorscale and not edge_colorbar else None
                ),
                "line": {"width": 1, "color": "black"},
            }

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
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
            x_coords = bif_positions[:, x_axis] * microns_per_voxel[x_axis]
            y_coords = bif_positions[:, y_axis] * microns_per_voxel[y_axis]

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    marker={
                        "size": 12,
                        "color": "yellow",
                        "symbol": "star",
                        "line": {"width": 2, "color": "black"},
                    },
                    name="Bifurcations",
                    hovertemplate="Bifurcation<extra></extra>",
                )
            )

        # Update layout
        fig.update_layout(**plot_2d_layout(projection_axis, x_label, y_label))
        # Ensure equal scaling so physical units are preserved
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig

    def plot_network_slice(
        self,
        vertices: dict[str, Any],
        edges: dict[str, Any],
        network: dict[str, Any],
        parameters: dict[str, Any],
        axis: int = 2,
        center_in_microns: float = 0.0,
        thickness_in_microns: float = 1.0,
        color_by: str = "energy",
        show_vertices: bool = True,
        show_edges: bool = True,
    ) -> go.Figure:
        """Create a 2D cross-sectional slice of the network.

        Parameters
        ----------
        vertices : Dict[str, Any]
            Vertex data.
        edges : Dict[str, Any]
            Edge data.
        network : Dict[str, Any]
            Network data (used for strand coloring).
        parameters : Dict[str, Any]
            Processing parameters containing voxel size.
        axis : int, optional
            Axis along which to take the slice (0=Y, 1=X, 2=Z), by default 2.
        center_in_microns : float, optional
            Center of the slice in microns along the chosen axis, by default 0.0.
        thickness_in_microns : float, optional
            Thickness of the slice in microns, by default 1.0.
        color_by : str, optional
            Coloring scheme ("energy", "depth", "strand_id", "radius", "length"), by default "energy".
        show_vertices : bool, optional
            Whether to show vertices, by default True.
        show_edges : bool, optional
            Whether to show edges, by default True.

        Returns
        -------
        go.Figure
            2D Plotly figure showing network elements within the slice.
        """

        microns_per_voxel = parameters.get("microns_per_voxel", [1.0, 1.0, 1.0])
        slice_min = center_in_microns - thickness_in_microns / 2.0
        slice_max = center_in_microns + thickness_in_microns / 2.0

        axes = [0, 1, 2]
        axes.remove(axis)
        x_axis, y_axis = axes
        axis_names = ["Y", "X", "Z"]
        x_label = f"{axis_names[x_axis]} (μm)"
        y_label = f"{axis_names[y_axis]} (μm)"

        fig = go.Figure()

        edge_traces = edges.get("traces", [])

        # Pre-compute strand IDs if needed
        strand_ids: list[int] = []
        if color_by == "strand_id":
            connections = edges.get("connections", [])
            pair_to_index = {
                tuple(sorted(map(int, conn))): idx for idx, conn in enumerate(connections)
            }
            strand_ids = [-1] * len(edge_traces)
            for sid, strand in enumerate(network.get("strands", [])):
                for v0, v1 in zip(strand[:-1], strand[1:]):
                    idx = pair_to_index.get(tuple(sorted((int(v0), int(v1)))))
                    if idx is not None:
                        strand_ids[idx] = sid
        else:
            strand_ids = []

        if show_edges and edge_traces:
            for i, trace in enumerate(edge_traces):
                arr = np.asarray(trace) * microns_per_voxel
                mask = (arr[:, axis] >= slice_min) & (arr[:, axis] <= slice_max)
                if np.count_nonzero(mask) < 2:
                    continue

                x_coords = arr[mask, x_axis]
                y_coords = arr[mask, y_axis]

                # Determine color
                # Determine color
                if color_by == "depth":
                    # To properly map a single value, we need min/max of the whole set.
                    # However, calculating it per edge is inefficient and technically valid if we assume
                    # the user wants relative coloring. But with single element, it fails in _map_values_to_colors.
                    # Better approach: map value using fixed range if we had one, or handle single value case.
                    # Here we'll just fix the single-value mapping issue by checking vmin/vmax logic in _map.
                    # But _map_values_to_colors takes an array.
                    # Let's use a workaround: pass [val, val] or fix _map...
                    # The issue description says: "Collect all values... then map".
                    # That requires two passes.
                    # Let's do a simple fix: use the slice bounds for depth? No, that's too narrow.
                    # Let's use the whole network depth range.
                    all_depths = vertices["positions"][:, axis] * microns_per_voxel[axis]
                    vmin, vmax = np.min(all_depths), np.max(all_depths)
                    depth = float(np.mean(arr[:, axis]))

                    # Normalize manually
                    norm = (depth - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    color = px.colors.sample_colorscale(self.color_schemes["depth"], norm)[0]

                elif color_by == "energy":
                    energies = edges.get("energies", [])
                    color = "blue"
                    if len(energies) == len(edge_traces):
                        all_energies = np.array(energies)
                        vmin, vmax = np.nanmin(all_energies), np.nanmax(all_energies)
                        val = energies[i]
                        norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                        color = px.colors.sample_colorscale(self.color_schemes["energy"], norm)[0]
                elif color_by == "radius":
                    connections = edges.get("connections", [])
                    radii = vertices.get("radii_microns", vertices.get("radii", []))
                    color = "blue"
                    if len(connections) == len(edge_traces) and len(radii) > 0:
                        v0, v1 = connections[i]
                        r0 = radii[int(v0)] if int(v0) >= 0 else 0
                        r1 = radii[int(v1)] if int(v1) >= 0 and int(v1) < len(radii) else r0
                        val = (r0 + r1) / 2.0
                        vmin, vmax = np.nanmin(radii), np.nanmax(radii)
                        norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                        color = px.colors.sample_colorscale(self.color_schemes["radius"], norm)[0]
                elif color_by == "strand_id":
                    sid = strand_ids[i] if strand_ids else -1
                    colors = px.colors.qualitative.Set3
                    color = colors[sid % len(colors)] if sid >= 0 else "blue"
                else:
                    color = "blue"

                name = f"Edge {i}" if i < 10 else ""
                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode="lines",
                        line={"color": color, "width": 2},
                        name=name,
                        showlegend=i < 10,
                        hovertemplate=(
                            f"Edge {i}<br>Length: {calculate_path_length(arr[mask]):.1f} μm<extra></extra>"
                        ),
                    )
                )

        if show_vertices and len(vertices.get("positions", [])) > 0:
            positions = vertices["positions"] * microns_per_voxel
            mask = (positions[:, axis] >= slice_min) & (positions[:, axis] <= slice_max)
            if np.any(mask):
                x_coords = positions[mask, x_axis]
                y_coords = positions[mask, y_axis]
                vertex_energies = vertices["energies"][mask]
                vertex_radii = vertices.get("radii_microns", vertices.get("radii", []))
                vertex_radii = vertex_radii[mask] if len(vertex_radii) > 0 else []

                if color_by == "energy":
                    colors = map_values_to_colors(vertex_energies, self.color_schemes["energy"])
                elif color_by == "depth":
                    depths = positions[mask, axis]
                    colors = map_values_to_colors(depths, self.color_schemes["depth"])
                elif color_by == "radius" and len(vertex_radii) > 0:
                    colors = map_values_to_colors(vertex_radii, self.color_schemes["radius"])
                else:
                    colors = ["red"] * len(x_coords)

                marker_dict = {
                    "size": 8,
                    "color": colors,
                    "line": {"width": 1, "color": "black"},
                }

                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode="markers",
                        marker=marker_dict,
                        name="Vertices",
                    )
                )

        fig.update_layout(**plot_slice_layout(center_in_microns, axis, x_label, y_label))
        # Ensure equal scaling between axes to avoid distortion
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig

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
                "colorbar": (
                    {"title": color_by.title()} if colorscale and not edge_colorbar else None
                ),
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

    def plot_energy_field(
        self, energy_data: dict[str, Any], slice_axis: int = 2, slice_index: int | None = None
    ) -> go.Figure:
        """
        Visualize energy field as 2D slice
        """
        logger.info("Creating energy field visualization")

        energy = energy_data["energy"]

        if slice_index is None:
            slice_index = energy.shape[slice_axis] // 2

        # Extract slice
        if slice_axis == 0:  # Y slice
            energy_slice = energy[slice_index, :, :]
            x_label, y_label = "X", "Z"
        elif slice_axis == 1:  # X slice
            energy_slice = energy[:, slice_index, :]
            x_label, y_label = "Y", "Z"
        else:  # Z slice
            energy_slice = energy[:, :, slice_index]
            x_label, y_label = "Y", "X"

        fig = go.Figure(
            data=go.Heatmap(
                z=energy_slice,
                colorscale="RdBu_r",
                colorbar={"title": "Energy"},
                hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>Energy: %{{z:.3f}}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Energy Field ({['Y', 'X', 'Z'][slice_axis]} slice at index {slice_index})",
            xaxis_title=f"{x_label} (voxels)",
            yaxis_title=f"{y_label} (voxels)",
            width=600,
            height=500,
        )

        return fig

    def plot_strand_analysis(
        self, network: dict[str, Any], vertices: dict[str, Any], parameters: dict[str, Any]
    ) -> go.Figure:
        """
        Create strand length and connectivity analysis
        """
        logger.info("Creating strand analysis plot")

        strands = network["strands"]
        vertex_positions = vertices["positions"]
        microns_per_voxel = parameters.get("microns_per_voxel", [1.0, 1.0, 1.0])

        # Calculate strand lengths
        strand_lengths = []
        for strand in strands:
            if len(strand) > 1:
                length = 0
                for i in range(len(strand) - 1):
                    pos1 = vertex_positions[strand[i]] * microns_per_voxel
                    pos2 = vertex_positions[strand[i + 1]] * microns_per_voxel
                    length += np.linalg.norm(pos2 - pos1)
                strand_lengths.append(length)

        if not strand_lengths:
            return empty_figure("No strands found")

        # Create histogram
        fig = go.Figure(
            data=go.Histogram(
                x=strand_lengths,
                nbinsx=min(20, len(strand_lengths)),
                name="Strand Lengths",
                hovertemplate="Length: %{x:.1f} μm<br>Count: %{y}<extra></extra>",
            )
        )

        # Add statistics
        mean_length = np.mean(strand_lengths)
        median_length = np.median(strand_lengths)

        fig.add_vline(
            x=mean_length,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_length:.1f} μm",
        )
        fig.add_vline(
            x=median_length,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Median: {median_length:.1f} μm",
        )

        fig.update_layout(
            **distribution_layout(
                "Strand Length Distribution",
                xaxis_title="Length (μm)",
                yaxis_title="Count",
                showlegend=True,
                width=600,
                height=400,
            )
        )
        return fig

    def plot_depth_statistics(
        self,
        vertices: dict[str, Any],
        edges: dict[str, Any],
        parameters: dict[str, Any],
        n_bins: int = 10,
    ) -> go.Figure:
        """
        Create depth-resolved statistics
        """
        logger.info("Creating depth statistics plot")

        vertex_positions = vertices["positions"]
        edge_traces = edges["traces"]
        microns_per_voxel = parameters.get("microns_per_voxel", [1.0, 1.0, 1.0])

        # Get Z coordinates (depth)
        vertex_depths = vertex_positions[:, 2] * microns_per_voxel[2]

        # Calculate edge lengths per depth bin
        edge_depths = []
        edge_lengths = []

        for trace in edge_traces:
            if len(trace) < 2:
                continue
            trace = np.array(trace)
            depth = np.mean(trace[:, 2]) * microns_per_voxel[2]
            length = calculate_path_length(trace * microns_per_voxel)
            edge_depths.append(depth)
            edge_lengths.append(length)

        if not edge_depths:
            return empty_figure("No edges found")

        # Create depth bins
        min_depth = min(min(vertex_depths), min(edge_depths))
        max_depth = max(max(vertex_depths), max(edge_depths))
        depth_bins = np.linspace(min_depth, max_depth, n_bins + 1)
        bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2

        # Bin vertices
        vertex_counts, _ = np.histogram(vertex_depths, bins=depth_bins)

        # Bin edge lengths
        length_sums, _ = np.histogram(edge_depths, bins=depth_bins, weights=edge_lengths)

        # Create subplot
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add vertex count
        fig.add_trace(
            go.Bar(x=bin_centers, y=vertex_counts, name="Vertex Count", opacity=0.7),
            secondary_y=False,
        )

        # Add total length
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=length_sums,
                name="Total Length",
                mode="lines+markers",
                line={"color": "red"},
            ),
            secondary_y=True,
        )

        # Update axes
        fig.update_xaxes(title_text="Depth (μm)")
        fig.update_yaxes(title_text="Vertex Count", secondary_y=False)
        fig.update_yaxes(title_text="Total Length (μm)", secondary_y=True)

        fig.update_layout(
            **distribution_layout(
                "Depth-Resolved Network Statistics",
                xaxis_title="Depth (μm)",
                yaxis_title="Vertex Count",
                showlegend=True,
                width=700,
                height=400,
            )
        )
        return fig

    def plot_radius_distribution(self, vertices: dict[str, Any]) -> go.Figure:
        """
        Create vessel radius distribution plot
        """
        logger.info("Creating radius distribution plot")

        radii = vertices.get("radii_microns", vertices.get("radii", []))

        fig = go.Figure(
            data=go.Histogram(
                x=radii,
                nbinsx=min(25, len(radii)),
                name="Vessel Radii",
                hovertemplate="Radius: %{x:.2f} μm<br>Count: %{y}<extra></extra>",
            )
        )

        # Add statistics
        mean_radius = np.mean(radii)
        median_radius = np.median(radii)

        fig.add_vline(
            x=mean_radius,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_radius:.2f} μm",
        )
        fig.add_vline(
            x=median_radius,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Median: {median_radius:.2f} μm",
        )

        fig.update_layout(
            **distribution_layout(
                "Vessel Radius Distribution",
                xaxis_title="Radius (μm)",
                yaxis_title="Count",
                width=600,
                height=400,
            )
        )
        return fig

    def plot_degree_distribution(self, network: dict[str, Any]) -> go.Figure:
        """
        Create vertex degree distribution plot
        """
        logger.info("Creating degree distribution plot")

        vertex_degrees = network.get("vertex_degrees", [])

        if len(vertex_degrees) == 0:
            return empty_figure("No degree data available")

        # Count degrees
        unique_degrees, counts = np.unique(vertex_degrees, return_counts=True)

        fig = go.Figure(
            data=go.Bar(
                x=unique_degrees,
                y=counts,
                name="Degree Distribution",
                hovertemplate="Degree: %{x}<br>Count: %{y}<extra></extra>",
            )
        )

        fig.update_layout(
            **distribution_layout(
                "Vertex Degree Distribution",
                xaxis_title="Degree",
                yaxis_title="Count",
                width=500,
                height=400,
            )
        )
        return fig

    def create_summary_dashboard(self, processing_results: dict[str, Any]) -> go.Figure:
        """
        Create comprehensive summary dashboard
        """
        logger.info("Creating summary dashboard")

        vertices = processing_results["vertices"]
        processing_results["edges"]
        network = processing_results["network"]
        parameters = processing_results["parameters"]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Network Overview",
                "Strand Lengths",
                "Radius Distribution",
                "Depth Statistics",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "histogram"}, {"secondary_y": True}],
            ],
        )

        add_summary_dashboard_traces(fig, vertices, network, parameters)

        fig.update_layout(**summary_dashboard_layout())
        return fig

    def export_network_data(
        self, processing_results: dict[str, Any], output_path: str, format: str = "csv"
    ) -> str:
        """
        Export network data in various formats

        Args:
            processing_results: Complete SLAVV processing results
            output_path: Output file path
            format: Export format ('csv', 'json', 'vmv', 'casx')

        Returns:
            Path to exported file
        """
        logger.info(f"Exporting network data in {format} format")

        vertices = processing_results["vertices"]
        edges = processing_results["edges"]
        network = processing_results["network"]
        parameters = processing_results["parameters"]

        if format == "csv":
            return export_csv(vertices, edges, network, parameters, output_path)
        if format == "json":
            return export_json(processing_results, output_path)
        if format == "vmv":
            return export_vmv(vertices, edges, network, parameters, output_path)
        if format == "casx":
            return export_casx(vertices, edges, network, parameters, output_path)
        if format == "mat":
            return export_mat(vertices, edges, network, parameters, output_path)
        raise ValueError(f"Unsupported export format: {format}")

    def plot_length_weighted_histograms(
        self,
        vertices: dict[str, Any],
        edges: dict[str, Any],
        parameters: dict[str, Any],
        number_of_bins: int = 50,
    ) -> go.Figure:
        """Create length-weighted histograms of radius, depth, and inclination.

        Port of MATLAB's area_histogram_plotter.m. Converts edge fragments
        into lengths, computes median/mean properties per segment, and weights
        the histogram bins by length (in mm).

        Parameters
        ----------
        vertices : Dict[str, Any]
            Vertex data containing positions and radii.
        edges : Dict[str, Any]
            Edge data containing traces and connections.
        parameters : Dict[str, Any]
            Processing parameters containing voxel size.
        number_of_bins : int, optional
            Number of bins for histograms, by default 50.

        Returns
        -------
        go.Figure
            Multipane Plotly figure containing the three histograms.
        """
        microns_per_voxel = np.array(parameters.get("microns_per_voxel", [1.0, 1.0, 1.0]))

        lengths_mm = []
        depths = []
        radii = []
        inclinations = []

        v_radii = vertices.get("radii_microns", vertices.get("radii", []))
        connections = edges.get("connections", [])

        for i, trace in enumerate(edges.get("traces", [])):
            trace_arr = np.array(trace)
            if len(trace_arr) < 2:
                continue

            # physical units
            trace_phys = trace_arr * microns_per_voxel

            # segment differences
            diffs = np.diff(trace_phys, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)

            # segment depth (Z-axis is index 2)
            seg_depths = (trace_phys[:-1, 2] + trace_phys[1:, 2]) / 2.0

            # inclination (abs dz / length) => [0, 1] component
            valid = seg_lengths > 1e-6
            seg_incls = np.zeros_like(seg_lengths)
            seg_incls[valid] = np.abs(diffs[valid, 2]) / seg_lengths[valid]

            # radius
            r0, r1 = 0.0, 0.0
            if len(v_radii) > 0 and i < len(connections):
                v0, v1 = int(connections[i][0]), int(connections[i][1])
                if 0 <= v0 < len(v_radii):
                    r0 = v_radii[v0]
                if 0 <= v1 < len(v_radii):
                    r1 = v_radii[v1]

            # interpolate radius along trace
            cum_len = np.insert(np.cumsum(seg_lengths), 0, 0.0)
            total_len = cum_len[-1]
            if total_len > 1e-6:
                seg_mids = (cum_len[:-1] + cum_len[1:]) / 2.0
                seg_rads = r0 + (r1 - r0) * (seg_mids / total_len)
            else:
                seg_rads = np.full_like(seg_lengths, (r0 + r1) / 2.0)

            lengths_mm.extend(seg_lengths / 1000.0)  # convert to mm for plotting
            depths.extend(seg_depths)
            radii.extend(seg_rads)
            inclinations.extend(seg_incls)

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Depth Distribution",
                "Radius Distribution (log10)",
                "Z-Axis Alignment",
            ],
            horizontal_spacing=0.1,
        )

        if lengths_mm:
            lengths_arr = np.array(lengths_mm)
            depths_arr = np.array(depths)
            radii_arr = np.array(radii)
            inclinations_arr = np.array(inclinations)

            # Subplot 1: Depth
            fig.add_trace(
                go.Histogram(
                    x=depths_arr,
                    y=lengths_arr,
                    histfunc="sum",
                    nbinsx=number_of_bins,
                    name="Depth",
                    marker_color="teal",
                    hovertemplate="Depth: %{x:.1f} μm<br>Length: %{y:.2f} mm<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Subplot 2: Radius (log10 x-axis)
            valid_r = radii_arr > 0
            log_radii = np.log10(radii_arr[valid_r])
            fig.add_trace(
                go.Histogram(
                    x=log_radii,
                    y=lengths_arr[valid_r],
                    histfunc="sum",
                    nbinsx=number_of_bins,
                    name="Radius",
                    marker_color="coral",
                    hovertemplate="Log10 Radius: %{x:.2f}<br>Length: %{y:.2f} mm<extra></extra>",
                ),
                row=1,
                col=2,
            )

            # Subplot 3: Inclination
            fig.add_trace(
                go.Histogram(
                    x=inclinations_arr,
                    y=lengths_arr,
                    histfunc="sum",
                    nbinsx=number_of_bins,
                    name="Alignment",
                    marker_color="mediumpurple",
                    hovertemplate="Alignment [0-1]: %{x:.2f}<br>Length: %{y:.2f} mm<extra></extra>",
                ),
                row=1,
                col=3,
            )

        # Layout refinements
        fig.update_xaxes(title_text="Depth (μm)", row=1, col=1)
        fig.update_yaxes(title_text="Total Length (mm)", row=1, col=1)

        fig.update_xaxes(title_text="Radius (μm) [10^x]", row=1, col=2)
        fig.update_yaxes(title_text="Total Length (mm)", row=1, col=2)

        fig.update_xaxes(title_text="Orientation [Component vs Z]", range=[0, 1], row=1, col=3)
        fig.update_yaxes(title_text="Total Length (mm)", row=1, col=3)

        fig.update_layout(
            title_text="Length-Weighted Histograms",
            showlegend=False,
            height=400,
            width=1000,
            bargap=0.05,
        )

        return fig
