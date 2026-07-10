"""NetworkVisualizer: deep facade for network plotting and export.

Owns color policy and stage-payload normalization so plot modules
(``spatial_2d``, ``spatial_3d``, ``statistics``) remain free of Stage Result
vs dict adaptation and theme defaults.

Plot implementations live in sibling modules; this module is the public
interface and the place that absorbs Stage Result / Mapping adaptation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from slavv_python.visualization.network_plots.helpers import NETWORK_COLOR_SCHEMES
from slavv_python.visualization.network_plots.spatial_2d import (
    plot_2d_network as _plot_2d_network_impl,
)
from slavv_python.visualization.network_plots.spatial_2d import (
    plot_network_slice as _plot_network_slice_impl,
)
from slavv_python.visualization.network_plots.spatial_3d import (
    animate_strands_3d as _animate_strands_3d_impl,
)
from slavv_python.visualization.network_plots.spatial_3d import (
    plot_3d_network as _plot_3d_network_impl,
)
from slavv_python.visualization.network_plots.spatial_3d import (
    plot_flow_field as _plot_flow_field_impl,
)
from slavv_python.visualization.network_plots.statistics import (
    create_summary_dashboard as _create_summary_dashboard_impl,
)
from slavv_python.visualization.network_plots.statistics import (
    export_network_data as _export_network_data_impl,
)
from slavv_python.visualization.network_plots.statistics import (
    plot_degree_distribution as _plot_degree_distribution_impl,
)
from slavv_python.visualization.network_plots.statistics import (
    plot_depth_statistics as _plot_depth_statistics_impl,
)
from slavv_python.visualization.network_plots.statistics import (
    plot_energy_field as _plot_energy_field_impl,
)
from slavv_python.visualization.network_plots.statistics import (
    plot_length_weighted_histograms as _plot_length_weighted_histograms_impl,
)
from slavv_python.visualization.network_plots.statistics import (
    plot_radius_distribution as _plot_radius_distribution_impl,
)
from slavv_python.visualization.network_plots.statistics import (
    plot_strand_analysis as _plot_strand_analysis_impl,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

StageLike = Any  # Mapping or Stage Result with ``to_dict()``


def stage_payload(value: StageLike | None) -> dict[str, Any]:
    """Normalize a typed Stage Result or legacy mapping into a plot dict shell.

    Accepts ``None`` (empty dict), objects with ``to_dict()`` (EnergyResult,
    VertexSet, EdgeSet, NetworkResult, PipelineResult), or mappings.
    """
    if value is None:
        return {}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
        raise TypeError(f"to_dict() returned non-mapping: {type(payload)!r}")
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        "Expected a Stage Result with to_dict() or a Mapping; "
        f"got {type(value)!r}"
    )


def processing_payload(value: StageLike | None) -> dict[str, Any]:
    """Normalize a full processing/results envelope for dashboard and export."""
    return stage_payload(value)


class NetworkVisualizer:
    """Deep visualization facade for vascular Stage Results and exports.

    Owns:
    * color schemes (theme policy)
    * default ``color_by``
    * Stage Result → dict adaptation for plot modules

    Plot math and layout live in ``spatial_*`` / ``statistics`` / ``exports``.
    """

    def __init__(
        self,
        *,
        color_schemes: Mapping[str, str] | None = None,
        default_color_by: str = "energy",
    ) -> None:
        self.color_schemes: dict[str, str] = dict(color_schemes or NETWORK_COLOR_SCHEMES)
        self.default_color_by = str(default_color_by)

    def resolve_color_by(self, color_by: str | None = None) -> str:
        """Return an explicit color key, falling back to the facade default."""
        key = self.default_color_by if color_by is None else str(color_by)
        if key not in self.color_schemes:
            return self.default_color_by
        return key

    def plot_2d_network(
        self,
        vertices: StageLike,
        edges: StageLike,
        network: StageLike,
        parameters: Mapping[str, Any] | None,
        color_by: str | None = None,
        projection_axis: int = 2,
        show_vertices: bool = True,
        show_edges: bool = True,
        show_bifurcations: bool = True,
    ) -> go.Figure:
        return _plot_2d_network_impl(
            self.color_schemes,
            stage_payload(vertices),
            stage_payload(edges),
            stage_payload(network),
            dict(parameters or {}),
            color_by=self.resolve_color_by(color_by),
            projection_axis=projection_axis,
            show_vertices=show_vertices,
            show_edges=show_edges,
            show_bifurcations=show_bifurcations,
        )

    def plot_network_slice(
        self,
        vertices: StageLike,
        edges: StageLike,
        network: StageLike,
        parameters: Mapping[str, Any] | None,
        axis: int = 2,
        center_in_microns: float = 0.0,
        thickness_in_microns: float = 1.0,
        color_by: str | None = None,
        show_vertices: bool = True,
        show_edges: bool = True,
    ) -> go.Figure:
        return _plot_network_slice_impl(
            self.color_schemes,
            stage_payload(vertices),
            stage_payload(edges),
            stage_payload(network),
            dict(parameters or {}),
            axis=axis,
            center_in_microns=center_in_microns,
            thickness_in_microns=thickness_in_microns,
            color_by=self.resolve_color_by(color_by),
            show_vertices=show_vertices,
            show_edges=show_edges,
        )

    def plot_3d_network(
        self,
        vertices: StageLike,
        edges: StageLike,
        network: StageLike,
        parameters: Mapping[str, Any] | None,
        color_by: str | None = None,
        show_vertices: bool = True,
        show_edges: bool = True,
        show_bifurcations: bool = True,
        opacity_by: str | None = None,
    ) -> go.Figure:
        return _plot_3d_network_impl(
            self,
            stage_payload(vertices),
            stage_payload(edges),
            stage_payload(network),
            dict(parameters or {}),
            color_by=self.resolve_color_by(color_by),
            show_vertices=show_vertices,
            show_edges=show_edges,
            show_bifurcations=show_bifurcations,
            opacity_by=opacity_by,
        )

    def animate_strands_3d(
        self,
        vertices: StageLike,
        edges: StageLike,
        network: StageLike,
        parameters: Mapping[str, Any] | None,
    ) -> go.Figure:
        return _animate_strands_3d_impl(
            self,
            stage_payload(vertices),
            stage_payload(edges),
            stage_payload(network),
            dict(parameters or {}),
        )

    def plot_flow_field(
        self,
        edges: StageLike,
        parameters: Mapping[str, Any] | None,
    ) -> go.Figure:
        return _plot_flow_field_impl(self, stage_payload(edges), dict(parameters or {}))

    def plot_energy_field(
        self,
        energy_data: StageLike,
        slice_axis: int = 2,
        slice_index: int | None = None,
    ) -> go.Figure:
        return _plot_energy_field_impl(
            self,
            stage_payload(energy_data),
            slice_axis=slice_axis,
            slice_index=slice_index,
        )

    def plot_strand_analysis(
        self,
        network: StageLike,
        vertices: StageLike,
        parameters: Mapping[str, Any] | None,
    ) -> go.Figure:
        return _plot_strand_analysis_impl(
            self,
            stage_payload(network),
            stage_payload(vertices),
            dict(parameters or {}),
        )

    def plot_depth_statistics(
        self,
        vertices: StageLike,
        edges: StageLike,
        parameters: Mapping[str, Any] | None,
        n_bins: int = 10,
    ) -> go.Figure:
        return _plot_depth_statistics_impl(
            self,
            stage_payload(vertices),
            stage_payload(edges),
            dict(parameters or {}),
            n_bins=n_bins,
        )

    def plot_radius_distribution(self, vertices: StageLike) -> go.Figure:
        return _plot_radius_distribution_impl(self, stage_payload(vertices))

    def plot_degree_distribution(self, network: StageLike) -> go.Figure:
        return _plot_degree_distribution_impl(self, stage_payload(network))

    def create_summary_dashboard(self, processing_results: StageLike) -> go.Figure:
        return _create_summary_dashboard_impl(self, processing_payload(processing_results))

    def export_network_data(
        self,
        processing_results: StageLike,
        output_path: str,
        format: str = "csv",
    ) -> str:
        return cast(
            "str",
            _export_network_data_impl(
                self,
                processing_payload(processing_results),
                output_path,
                format=format,
            ),
        )

    def plot_length_weighted_histograms(
        self,
        vertices: StageLike,
        edges: StageLike,
        parameters: Mapping[str, Any] | None,
        number_of_bins: int = 50,
    ) -> go.Figure:
        return _plot_length_weighted_histograms_impl(
            self,
            stage_payload(vertices),
            stage_payload(edges),
            dict(parameters or {}),
            number_of_bins=number_of_bins,
        )


__all__ = [
    "NetworkVisualizer",
    "processing_payload",
    "stage_payload",
]
