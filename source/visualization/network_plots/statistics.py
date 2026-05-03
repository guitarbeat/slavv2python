"""Statistics and export plotting helpers for SLAVV network visualizations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .dashboard import add_summary_dashboard_traces
from .exports import export_casx, export_csv, export_json, export_mat, export_vmv
from .layout import distribution_layout, empty_figure, summary_dashboard_layout
from ...models import normalize_pipeline_result
from ...utils import calculate_path_length

logger = logging.getLogger(__name__)


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

    typed_result = normalize_pipeline_result(processing_results)
    vertices = typed_result.vertices.to_dict() if typed_result.vertices else {}
    network = typed_result.network.to_dict() if typed_result.network else {}
    parameters = typed_result.parameters

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

    typed_result = normalize_pipeline_result(processing_results)
    normalized_results = typed_result.to_dict()
    vertices = normalized_results["vertices"]
    edges = normalized_results["edges"]
    network = normalized_results["network"]
    parameters = typed_result.parameters

    if format == "csv":
        return export_csv(vertices, edges, network, parameters, output_path)
    if format == "json":
        return export_json(normalized_results, output_path)
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
