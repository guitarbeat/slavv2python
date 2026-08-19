"""Visualization page for the SLAVV Streamlit app."""

from __future__ import annotations

import streamlit as st

from slavv_python.interface.streamlit.empty_state import require_processing_results
from slavv_python.interface.streamlit.services import app as app_services
from slavv_python.interface.streamlit.services.share_report import record_share_event
from slavv_python.interface.streamlit.state.visualization import (
    extract_visualization_export_payload,
    has_visualization_network,
    list_available_visualizations,
    normalize_visualization_results,
    resolve_visualization_session_context,
)
from slavv_python.visualization import NetworkVisualizer

EXPORT_BUTTON_SPECS = (
    {
        "format_type": "vmv",
        "label": "Download VMV",
        "empty_label": "Export VMV",
        "file_name": "network.vmv",
        "mime": "text/plain",
        "help": "Export network in VessMorphoVis (VMV) format",
        "artifact_key": "vmv_file",
    },
    {
        "format_type": "casx",
        "label": "Download CASX",
        "empty_label": "Export CASX",
        "file_name": "network.casx",
        "mime": "application/xml",
        "help": "Export network in CASX XML format",
        "artifact_key": "casx_file",
    },
    {
        "format_type": "csv",
        "label": "Download CSV (Zip)",
        "empty_label": "Export CSV",
        "file_name": "network_csv.zip",
        "mime": "application/zip",
        "help": "Export network data as Zipped CSVs (vertices & edges)",
        "artifact_key": "csv_archive",
    },
)


def _render_export_download(
    column,
    *,
    run_dir: str | None,
    vertices,
    edges,
    network,
    parameters,
    export_spec: dict[str, str],
    generate_export_data_fn=None,
    update_run_task_fn=None,
) -> None:
    """Render one export button using a shared table-driven config."""
    if generate_export_data_fn is None or update_run_task_fn is None:
        generate_export_data_fn = app_services.generate_export_data
        update_run_task_fn = app_services._update_run_task

    with column:
        export_data = generate_export_data_fn(
            vertices,
            edges,
            network,
            parameters,
            export_spec["format_type"],
        )
        if export_data:
            update_run_task_fn(
                run_dir,
                "exports",
                status="completed",
                detail="App export downloads prepared",
                artifacts={export_spec["artifact_key"]: export_spec["file_name"]},
            )
            st.download_button(
                label=export_spec["label"],
                data=export_data,
                file_name=export_spec["file_name"],
                mime=export_spec["mime"],
                help=export_spec["help"],
            )
        else:
            st.button(
                export_spec["empty_label"],
                disabled=True,
                help="Export generation failed",
            )


_CAMERA_EYE = {
    "Isometric": {"x": 1.6, "y": 1.6, "z": 1.2},
    "Top": {"x": 0.0, "y": 0.0, "z": 2.4},
    "Side": {"x": 2.4, "y": 0.0, "z": 0.2},
    "Front": {"x": 0.0, "y": 2.4, "z": 0.2},
}


def _apply_figure_display(fig, *, opacity: float, camera: str | None = None) -> None:
    """Apply opacity and an optional 3D camera eye to a Plotly figure."""
    fig.update_traces(opacity=float(opacity))
    if camera is not None:
        fig.update_layout(scene_camera={"eye": _CAMERA_EYE[camera]})


def show_visualization_page() -> None:
    """Display the visualization page."""
    st.markdown('<h2 class="section-header">Network Visualization</h2>', unsafe_allow_html=True)

    results = require_processing_results()
    if results is None:
        return

    st.markdown(
        "Explore the Energy field and the vectorized Network in 2D or 3D. "
        "Exports (VMV, CASX, CSV zip, share HTML) unlock after the Network stage."
    )

    results = normalize_visualization_results(st.session_state["processing_results"])
    available_viz = list_available_visualizations(results)

    if not available_viz:
        st.warning("No visualizable results found in the current run.")
        return

    viz_type = st.selectbox(
        "Visualization type",
        available_viz,
        help="Choose the type of visualization to display",
    )
    col1, col2 = st.columns([3, 1], gap="large")

    with col2:
        st.markdown("### Display Options")
        show_vertices = True
        show_edges = True
        show_bifurcations = True
        color_scheme = "Energy"
        opacity = 0.8
        camera: str | None = None
        slice_axis = 0
        slice_index = 0
        if viz_type == "Energy Field":
            energy = results["energy_data"]["energy"]
            slice_axis = st.selectbox(
                "Slice axis",
                [0, 1, 2],
                format_func=lambda x: ["Y", "X", "Z"][x],
                help="Which volume axis to cut.",
            )
            max_idx = max(int(energy.shape[slice_axis]) - 1, 0)
            slice_index = int(
                st.number_input(
                    "Slice index",
                    min_value=0,
                    max_value=max_idx,
                    value=max_idx // 2,
                    help="Index along the selected axis.",
                )
            )
        else:
            show_vertices = st.checkbox(
                "Show vertices", value=True, help="Display detected vertex markers"
            )
            show_edges = st.checkbox(
                "Show edges", value=True, help="Display vessel segments from the Edge Set"
            )
            show_bifurcations = st.checkbox(
                "Show bifurcations", value=True, help="Highlight branching points in the Network"
            )
            color_scheme = st.selectbox(
                "Color scheme",
                ["Energy", "Depth", "Strand ID", "Radius", "Length", "Random"],
                help="How to color the network components",
            )
            opacity = float(
                st.slider(
                    "Opacity",
                    0.1,
                    1.0,
                    0.8,
                    0.1,
                    help="Trace and marker transparency.",
                )
            )
            if viz_type == "3D Network":
                camera = st.selectbox(
                    "Camera angle",
                    ["Isometric", "Top", "Side", "Front"],
                    help="3D viewing angle",
                )

    visualizer = NetworkVisualizer()
    with col1:
        st.markdown(f"### {viz_type}")
        if viz_type == "2D Network":
            fig = visualizer.plot_2d_network(
                results["vertices"],
                results["edges"],
                results["network"],
                results["parameters"],
                color_by=color_scheme.lower().replace(" ", "_"),
                show_vertices=show_vertices,
                show_edges=show_edges,
                show_bifurcations=show_bifurcations,
            )
            _apply_figure_display(fig, opacity=opacity)
            st.plotly_chart(fig, width="stretch")
        elif viz_type == "3D Network":
            fig = visualizer.plot_3d_network(
                results["vertices"],
                results["edges"],
                results["network"],
                results["parameters"],
                color_by=color_scheme.lower().replace(" ", "_"),
                show_vertices=show_vertices,
                show_edges=show_edges,
                show_bifurcations=show_bifurcations,
            )
            _apply_figure_display(fig, opacity=opacity, camera=camera)
            st.plotly_chart(fig, width="stretch")
        elif viz_type == "Depth Projection":
            fig = visualizer.plot_depth_statistics(
                results["vertices"],
                results["edges"],
                results["parameters"],
            )
            _apply_figure_display(fig, opacity=opacity)
            st.plotly_chart(fig, width="stretch")
        elif viz_type == "Strand Analysis":
            fig = visualizer.plot_strand_analysis(
                results["network"],
                results["vertices"],
                results["parameters"],
            )
            _apply_figure_display(fig, opacity=opacity)
            st.plotly_chart(fig, width="stretch")
        elif viz_type == "Energy Field":
            st.info("Energy Field is a 2D slice through the Energy volume.")
            fig = visualizer.plot_energy_field(
                results["energy_data"],
                slice_axis=slice_axis,
                slice_index=slice_index,
            )
            st.plotly_chart(fig, width="stretch")

    if not has_visualization_network(results):
        st.info("Complete the full network stage to unlock exports and the share report.")
        return

    st.markdown("### Export Options")
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    vertices, edges, network, parameters = extract_visualization_export_payload(results)
    viz_context = resolve_visualization_session_context(st.session_state)
    current_run_dir = viz_context["run_dir"]
    for column, export_spec in zip((col1, col2, col3), EXPORT_BUTTON_SPECS):
        _render_export_download(
            column,
            run_dir=current_run_dir,
            vertices=vertices,
            edges=edges,
            network=network,
            parameters=parameters,
            export_spec=export_spec,
        )
    share_report_data = app_services.generate_share_report_data(
        results,
        viz_context["dataset_name"],
        viz_context["image_shape"],
    )
    app_services._log_share_report_prepared_once(
        viz_context["dataset_name"],
        share_report_data,
        results,
    )
    app_services._update_run_task(
        st.session_state.get("current_run_dir"),
        "share_report",
        status="completed",
        detail="Share report generated in app",
        artifacts={
            "share_report_file": share_report_data["file_name"],
            "share_report_signature": share_report_data["signature"],
        },
    )
    with col4:
        downloaded = st.download_button(
            label="Download Share Report",
            data=share_report_data["html"],
            file_name=share_report_data["file_name"],
            mime="text/html",
            help="Download a self-contained HTML report to share with collaborators.",
        )
        if downloaded:
            record_share_event(
                st.session_state,
                "share_report_downloaded",
                viz_context["dataset_name"],
                share_report_data["signature"],
                extra={"report_file_name": share_report_data["file_name"]},
            )
            app_services._update_run_task(
                st.session_state.get("current_run_dir"),
                "share_report",
                status="completed",
                detail="Share report downloaded",
                artifacts={"downloaded_report": share_report_data["file_name"]},
            )
    if downloaded:
        st.success(
            "Share report downloaded. Forward the HTML file to collaborators for offline review."
        )
    share_metrics = viz_context["share_metrics"]
    st.caption(
        "Tracked share events this session: "
        f"requested={share_metrics.get('share_report_requested', 0)}, "
        f"downloaded={share_metrics.get('share_report_downloaded', 0)}"
    )
