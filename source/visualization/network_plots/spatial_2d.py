"""2D spatial plotting helpers for SLAVV network visualizations."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from .helpers import add_colorbar, map_values_to_colors
from .layout import axis_labels, plot_2d_layout, plot_slice_layout, select_plot_axes
from ...utils import calculate_path_length


def plot_2d_network(
        color_schemes: dict[str, str],
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
    """Create 2D network visualization with projection."""
    fig = go.Figure()

    vertex_positions = vertices["positions"]
    vertex_energies = vertices["energies"]
    vertex_radii = vertices.get("radii_microns", vertices.get("radii", []))
    edge_traces = edges["traces"]
    bifurcations = network.get("bifurcations", [])

    x_axis, y_axis = select_plot_axes(projection_axis)
    x_label, y_label = axis_labels(x_axis, y_axis)
    microns_per_voxel = parameters.get("microns_per_voxel", [1.0, 1.0, 1.0])

    if show_edges and edge_traces:
        valid_traces = [np.array(t) for t in edge_traces if len(t) >= 2]
        edge_colors: list[str] = []
        strand_ids: list[int] = []
        values: np.ndarray | None = None

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
            lengths = [calculate_path_length(trace * microns_per_voxel) for trace in valid_traces]
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

        if values is not None:
            n_bins = 64
            vmin, vmax = np.min(values), np.max(values)
            if vmax > vmin:
                bins = np.linspace(vmin, vmax, n_bins)
                indices = np.searchsorted(bins, values)
                indices = np.clip(indices, 0, len(bins) - 1)
                quantized_values = bins[indices]
                edge_colors = map_values_to_colors(quantized_values, color_schemes[color_by])
            else:
                edge_colors = map_values_to_colors(values, color_schemes[color_by])
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
            edge_colors = [colors[sid % len(colors)] if sid >= 0 else "blue" for sid in strand_ids]
        else:
            edge_colors = ["blue"] * len(valid_traces)

        batched_traces: dict[str, dict[str, list]] = {}
        for i, trace in enumerate(valid_traces):
            color = edge_colors[i]
            if color not in batched_traces:
                batched_traces[color] = {"x": [], "y": [], "customdata": [], "names": []}

            xs = trace[:, x_axis] * microns_per_voxel[x_axis]
            ys = trace[:, y_axis] * microns_per_voxel[y_axis]
            batched_traces[color]["x"].extend(xs)
            batched_traces[color]["x"].append(None)
            batched_traces[color]["y"].extend(ys)
            batched_traces[color]["y"].append(None)

            length = calculate_path_length(trace * microns_per_voxel)
            if color_by == "strand_id":
                sid = strand_ids[i]
                name = f"Strand {sid}"
            else:
                name = f"Edge {i}"
            edge_meta = [i, length, name]
            batched_traces[color]["customdata"].extend([edge_meta] * len(xs))
            batched_traces[color]["customdata"].append([None, None, None])

        for color, data in batched_traces.items():
            fig.add_trace(
                go.Scattergl(
                    x=data["x"],
                    y=data["y"],
                    mode="lines",
                    line={"color": color, "width": 2},
                    name="Edges",
                    showlegend=False,
                    customdata=data["customdata"],
                    hovertemplate=(
                        "%{customdata[2]}<br>Length: %{customdata[1]:.1f} μm<extra></extra>"
                    ),
                )
            )

        if color_by in {"depth", "energy", "radius", "length"} and values is not None:
            add_colorbar(fig, values, color_schemes[color_by], color_by.title(), is_3d=False)

    if show_vertices and len(vertex_positions) > 0:
        x_coords = vertex_positions[:, x_axis] * microns_per_voxel[x_axis]
        y_coords = vertex_positions[:, y_axis] * microns_per_voxel[y_axis]

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
            "colorbar": {"title": color_by.title()} if colorscale and not edge_colorbar else None,
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

    fig.update_layout(**plot_2d_layout(projection_axis, x_label, y_label))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_network_slice(
        color_schemes: dict[str, str],
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
    """Create a 2D cross-sectional slice of the network."""
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

    strand_ids: list[int] = []
    if color_by == "strand_id":
        connections = edges.get("connections", [])
        pair_to_index = {tuple(sorted(map(int, conn))): idx for idx, conn in enumerate(connections)}
        strand_ids = [-1] * len(edge_traces)
        for sid, strand in enumerate(network.get("strands", [])):
            for v0, v1 in zip(strand[:-1], strand[1:]):
                idx = pair_to_index.get(tuple(sorted((int(v0), int(v1)))))
                if idx is not None:
                    strand_ids[idx] = sid

    if show_edges and edge_traces:
        for i, trace in enumerate(edge_traces):
            arr = np.asarray(trace) * microns_per_voxel
            mask = (arr[:, axis] >= slice_min) & (arr[:, axis] <= slice_max)
            if np.count_nonzero(mask) < 2:
                continue

            x_coords = arr[mask, x_axis]
            y_coords = arr[mask, y_axis]

            if color_by == "depth":
                all_depths = vertices["positions"][:, axis] * microns_per_voxel[axis]
                vmin, vmax = np.min(all_depths), np.max(all_depths)
                depth = float(np.mean(arr[:, axis]))
                norm = (depth - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                color = px.colors.sample_colorscale(color_schemes["depth"], norm)[0]
            elif color_by == "energy":
                energies = edges.get("energies", [])
                color = "blue"
                if len(energies) == len(edge_traces):
                    all_energies = np.array(energies)
                    vmin, vmax = np.nanmin(all_energies), np.nanmax(all_energies)
                    val = energies[i]
                    norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    color = px.colors.sample_colorscale(color_schemes["energy"], norm)[0]
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
                    color = px.colors.sample_colorscale(color_schemes["radius"], norm)[0]
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
                colors = map_values_to_colors(vertex_energies, color_schemes["energy"])
            elif color_by == "depth":
                depths = positions[mask, axis]
                colors = map_values_to_colors(depths, color_schemes["depth"])
            elif color_by == "radius" and len(vertex_radii) > 0:
                colors = map_values_to_colors(vertex_radii, color_schemes["radius"])
            else:
                colors = ["red"] * len(x_coords)

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    marker={"size": 8, "color": colors, "line": {"width": 1, "color": "black"}},
                    name="Vertices",
                )
            )

    fig.update_layout(**plot_slice_layout(center_in_microns, axis, x_label, y_label))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
