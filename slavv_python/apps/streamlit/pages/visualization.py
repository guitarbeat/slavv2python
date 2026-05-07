"""Visualization page for the SLAVV Streamlit app."""

from __future__ import annotations

import streamlit as st

from slavv_python.apps.services import app as app_services
from slavv_python.apps.state.visualization import (
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
        "label": "ðŸ“„ Download VMV",
        "empty_label": "ðŸ“„ Export VMV",
        "file_name": "network.vmv",
        "mime": "text/plain",
        "help": "Export network in VessMorphoVis (VMV) format",
        "artifact_key": "vmv_file",
    },
    {
        "format_type": "casx",
        "label": "ðŸ“„ Download CASX",
        "empty_label": "ðŸ“„ Export CASX",
        "file_name": "network.casx",
        "mime": "application/xml",
        "help": "Export network in CASX XML format",
        "artifact_key": "casx_file",
    },
    {
        "format_type": "csv",
        "label": "ðŸ“Š Download CSV (Zip)",
        "empty_label": "ðŸ“Š Export CSV",
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


def show_visualization_page() -> None:
    """Display the visualization page."""
    st.markdown('<h2 class="section-header">Network Visualization</h2>', unsafe_allow_html=True)

    if "processing_results" not in st.session_state:
        st.warning("No processing results found. Please process an image first.")
        return

    st.markdown(
        """
    Visualize the vectorized vascular network in 2D and 3D. This section provides interactive tools to explore the results.
    Corresponds to `Visual` and `SpecialOutput` parameters in MATLAB.
    """
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
        show_vertices = st.checkbox(
            "Show vertices", value=True, help="Display detected vertex markers"
        )
        show_edges = st.checkbox("Show edges", value=True, help="Display traced vessel segments")
        show_bifurcations = st.checkbox(
            "Show bifurcations", value=True, help="Highlight branching points in the network"
        )
        color_scheme = st.selectbox(
            "Color scheme",
            ["Energy", "Depth", "Strand ID", "Radius", "Length", "Random"],
            help="How to color the network components",
        )
        st.slider("Opacity", 0.1, 1.0, 0.8, 0.1, help="Adjust transparency of network rendering")
        if viz_type == "3D Network":
            st.selectbox(
                "Camera angle", ["Isometric", "Top", "Side", "Front"], help="3D viewing angle"
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
            st.plotly_chart(fig, use_container_width=True)
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
            st.plotly_chart(fig, use_container_width=True)
        elif viz_type == "Depth Projection":
            fig = visualizer.plot_depth_statistics(
                results["vertices"],
                results["edges"],
                results["parameters"],
            )
            st.plotly_chart(fig, use_container_width=True)
        elif viz_type == "Strand Analysis":
            fig = visualizer.plot_strand_analysis(
                results["network"],
                results["vertices"],
                results["parameters"],
            )
            st.plotly_chart(fig, use_container_width=True)
        elif viz_type == "Energy Field":
            st.info(
                "Energy Field visualization is a 2D slice. Select slice axis and index in sidebar."
            )
            slice_axis = st.sidebar.selectbox(
                "Slice Axis", [0, 1, 2], format_func=lambda x: ["Y", "X", "Z"][x]
            )
            energy = results["energy_data"]["energy"]
            slice_index = st.sidebar.number_input(
                "Slice Index", value=int(energy.shape[slice_axis] // 2)
            )
            fig = visualizer.plot_energy_field(
                results["energy_data"],
                slice_axis=slice_axis,
                slice_index=slice_index,
            )
            st.plotly_chart(fig, use_container_width=True)

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
            from ...services.share_report import record_share_event

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
