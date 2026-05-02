from __future__ import annotations

from source.visualization import NetworkVisualizer
from source.visualization.network_plot_layout import select_plot_axes as flat_select_plot_axes
from source.visualization.network_plots import NetworkVisualizer as PackageNetworkVisualizer
from source.visualization.network_plots.layout import select_plot_axes as package_select_plot_axes


def test_grouped_visualization_import_surfaces_export_expected_symbols():
    assert NetworkVisualizer is PackageNetworkVisualizer
    assert flat_select_plot_axes is package_select_plot_axes
