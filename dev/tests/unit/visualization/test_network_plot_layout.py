from source.visualization.network_plot_layout import (
    AXIS_NAMES,
    axis_labels,
    distribution_layout,
    empty_figure,
    plot_2d_layout,
    plot_3d_layout,
    plot_slice_layout,
    select_plot_axes,
    summary_dashboard_layout,
)

MICRON_LABEL = "\u03bcm"


def test_select_plot_axes_and_labels_cover_projection_convention():
    assert select_plot_axes(2) == (1, 0)
    assert select_plot_axes(1) == (0, 2)
    assert axis_labels(1, 0) == (f"X ({MICRON_LABEL})", f"Y ({MICRON_LABEL})")
    assert AXIS_NAMES == ("Y", "X", "Z")


def test_layout_helpers_return_expected_titles():
    layout = plot_2d_layout(2, f"X ({MICRON_LABEL})", f"Y ({MICRON_LABEL})")
    assert layout["title"] == "2D Vascular Network (Projection along Z)"
    assert layout["xaxis_title"] == f"X ({MICRON_LABEL})"
    assert layout["yaxis_title"] == f"Y ({MICRON_LABEL})"
    assert layout["width"] == 800
    assert layout["height"] == 600

    slice_layout = plot_slice_layout(12.5, 0, f"Y ({MICRON_LABEL})", f"Z ({MICRON_LABEL})")
    assert slice_layout["title"] == f"Network Slice at 12.5 {MICRON_LABEL} along Y"
    assert slice_layout["xaxis_title"] == f"Y ({MICRON_LABEL})"
    assert slice_layout["yaxis_title"] == f"Z ({MICRON_LABEL})"


def test_plot_3d_layout_supports_animation_controls():
    layout = plot_3d_layout(
        "Animated 3D Strands",
        showlegend=False,
        sliders=[{"active": 0}],
        updatemenus=[{"type": "buttons"}],
    )
    assert layout["title"] == "Animated 3D Strands"
    assert layout["scene"]["xaxis_title"] == f"X ({MICRON_LABEL})"
    assert layout["scene"]["aspectmode"] == "data"
    assert layout["showlegend"] is False
    assert layout["sliders"] == [{"active": 0}]
    assert layout["updatemenus"] == [{"type": "buttons"}]


def test_distribution_layout_supports_shared_axis_labels():
    layout = distribution_layout(
        "Depth-Resolved Network Statistics",
        xaxis_title=f"Depth ({MICRON_LABEL})",
        yaxis_title="Vertex Count",
        showlegend=True,
        width=700,
        height=400,
    )
    assert layout["title"] == "Depth-Resolved Network Statistics"
    assert layout["xaxis_title"] == f"Depth ({MICRON_LABEL})"
    assert layout["yaxis_title"] == "Vertex Count"
    assert layout["showlegend"] is True
    assert layout["width"] == 700
    assert layout["height"] == 400


def test_summary_dashboard_layout_returns_shared_dashboard_defaults():
    assert summary_dashboard_layout() == {
        "title": "SLAVV Processing Summary Dashboard",
        "showlegend": False,
        "height": 600,
        "width": 1000,
    }


def test_empty_figure_returns_centered_annotation():
    fig = empty_figure("No strands found")
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0].text == "No strands found"
    assert fig.layout.annotations[0].x == 0.5
    assert fig.layout.annotations[0].y == 0.5


