from slavv.visualization.network_plot_layout import (
    AXIS_NAMES,
    axis_labels,
    plot_2d_layout,
    plot_slice_layout,
    select_plot_axes,
)


def test_select_plot_axes_and_labels_cover_projection_convention():
    assert select_plot_axes(2) == (1, 0)
    assert select_plot_axes(1) == (0, 2)
    assert axis_labels(1, 0) == ("X (μm)", "Y (μm)")
    assert AXIS_NAMES == ("Y", "X", "Z")


def test_layout_helpers_return_expected_titles():
    layout = plot_2d_layout(2, "X (μm)", "Y (μm)")
    assert layout["title"] == "2D Vascular Network (Projection along Z)"
    assert layout["xaxis_title"] == "X (μm)"
    assert layout["yaxis_title"] == "Y (μm)"
    assert layout["width"] == 800
    assert layout["height"] == 600

    slice_layout = plot_slice_layout(12.5, 0, "Y (μm)", "Z (μm)")
    assert slice_layout["title"] == "Network Slice at 12.5 μm along Y"
    assert slice_layout["xaxis_title"] == "Y (μm)"
    assert slice_layout["yaxis_title"] == "Z (μm)"
