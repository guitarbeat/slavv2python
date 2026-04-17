"""
Visualization module for SLAVV results

This module provides comprehensive visualization capabilities for vascular networks
including 2D/3D plotting, statistical analysis, and export functionality.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .network_plot_helpers import NETWORK_COLOR_SCHEMES
from .network_plot_spatial_2d import plot_2d_network as _plot_2d_network_impl
from .network_plot_spatial_2d import plot_network_slice as _plot_network_slice_impl
from .network_plot_spatial_3d import animate_strands_3d as _animate_strands_3d_impl
from .network_plot_spatial_3d import plot_3d_network as _plot_3d_network_impl
from .network_plot_spatial_3d import plot_flow_field as _plot_flow_field_impl
from .network_plot_statistics import create_summary_dashboard as _create_summary_dashboard_impl
from .network_plot_statistics import export_network_data as _export_network_data_impl
from .network_plot_statistics import plot_degree_distribution as _plot_degree_distribution_impl
from .network_plot_statistics import plot_depth_statistics as _plot_depth_statistics_impl
from .network_plot_statistics import plot_energy_field as _plot_energy_field_impl
from .network_plot_statistics import (
    plot_length_weighted_histograms as _plot_length_weighted_histograms_impl,
)
from .network_plot_statistics import plot_radius_distribution as _plot_radius_distribution_impl
from .network_plot_statistics import plot_strand_analysis as _plot_strand_analysis_impl

if TYPE_CHECKING:
    import plotly.graph_objects as go

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
        return _plot_2d_network_impl(
            self.color_schemes,
            vertices,
            edges,
            network,
            parameters,
            color_by=color_by,
            projection_axis=projection_axis,
            show_vertices=show_vertices,
            show_edges=show_edges,
            show_bifurcations=show_bifurcations,
        )

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
        return _plot_network_slice_impl(
            self.color_schemes,
            vertices,
            edges,
            network,
            parameters,
            axis=axis,
            center_in_microns=center_in_microns,
            thickness_in_microns=thickness_in_microns,
            color_by=color_by,
            show_vertices=show_vertices,
            show_edges=show_edges,
        )

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
        return _plot_3d_network_impl(
            self,
            vertices,
            edges,
            network,
            parameters,
            color_by=color_by,
            show_vertices=show_vertices,
            show_edges=show_edges,
            show_bifurcations=show_bifurcations,
            opacity_by=opacity_by,
        )

    def animate_strands_3d(
        self,
        vertices: dict[str, Any],
        edges: dict[str, Any],
        network: dict[str, Any],
        parameters: dict[str, Any],
    ) -> go.Figure:
        return _animate_strands_3d_impl(self, vertices, edges, network, parameters)

    def plot_flow_field(
        self,
        edges: dict[str, Any],
        parameters: dict[str, Any],
    ) -> go.Figure:
        return _plot_flow_field_impl(self, edges, parameters)

    def plot_energy_field(
        self, energy_data: dict[str, Any], slice_axis: int = 2, slice_index: int | None = None
    ) -> go.Figure:
        return _plot_energy_field_impl(
            self, energy_data, slice_axis=slice_axis, slice_index=slice_index
        )

    def plot_strand_analysis(
        self, network: dict[str, Any], vertices: dict[str, Any], parameters: dict[str, Any]
    ) -> go.Figure:
        return _plot_strand_analysis_impl(self, network, vertices, parameters)

    def plot_depth_statistics(
        self,
        vertices: dict[str, Any],
        edges: dict[str, Any],
        parameters: dict[str, Any],
        n_bins: int = 10,
    ) -> go.Figure:
        return _plot_depth_statistics_impl(self, vertices, edges, parameters, n_bins=n_bins)

    def plot_radius_distribution(self, vertices: dict[str, Any]) -> go.Figure:
        return _plot_radius_distribution_impl(self, vertices)

    def plot_degree_distribution(self, network: dict[str, Any]) -> go.Figure:
        return _plot_degree_distribution_impl(self, network)

    def create_summary_dashboard(self, processing_results: dict[str, Any]) -> go.Figure:
        return _create_summary_dashboard_impl(self, processing_results)

    def export_network_data(
        self, processing_results: dict[str, Any], output_path: str, format: str = "csv"
    ) -> str:
        return _export_network_data_impl(self, processing_results, output_path, format=format)

    def plot_length_weighted_histograms(
        self,
        vertices: dict[str, Any],
        edges: dict[str, Any],
        parameters: dict[str, Any],
        number_of_bins: int = 50,
    ) -> go.Figure:
        return _plot_length_weighted_histograms_impl(
            self,
            vertices,
            edges,
            parameters,
            number_of_bins=number_of_bins,
        )
