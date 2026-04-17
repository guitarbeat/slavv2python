"""Visualization page for the SLAVV Streamlit app."""

from __future__ import annotations

import streamlit as st

from slavv.visualization import NetworkVisualizer

EXPORT_BUTTON_SPECS = (
    {
        "format_type": "vmv",
        "label": "📄 Download VMV",
        "empty_label": "📄 Export VMV",
        "file_name": "network.vmv",
        "mime": "text/plain",
        "help": "Export network in VessMorphoVis (VMV) format",
        "artifact_key": "vmv_file",
    },
    {
        "format_type": "casx",
        "label": "📄 Download CASX",
        "empty_label": "📄 Export CASX",
        "file_name": "network.casx",
        "mime": "application/xml",
        "help": "Export network in CASX XML format",
        "artifact_key": "casx_file",
    },
    {
        "format_type": "csv",
        "label": "📊 Download CSV (Zip)",
        "empty_label": "📊 Export CSV",
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
) -> None:
    """Render one export button using a shared table-driven config."""
    from slavv.apps import web_app as web_app_facade

    with column:
        if export_data := web_app_facade.generate_export_data(
            vertices,
            edges,
            network,
            parameters,
            export_spec["format_type"],
        ):
            web_app_facade._update_run_task(
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
    from slavv.apps import web_app as web_app_facade

    st.markdown('<h2 class="section-header">Network Visualization</h2>', unsafe_allow_html=True)

    if "processing_results" not in st.session_state:
        st.warning("⚠️ No processing results found. Please process an image first.")
        return

    st.markdown(
        """
    Visualize the vectorized vascular network in 2D and 3D. This section provides interactive tools to explore the results.
    Corresponds to `Visual` and `SpecialOutput` parameters in MATLAB.
    """
    )

    results = st.session_state["processing_results"]
    available_viz = []
    if "energy_data" in results:
        available_viz.append("Energy Field")
    if "vertices" in results and "edges" in results and "network" in results:
        available_viz.extend(["2D Network", "3D Network", "Depth Projection", "Strand Analysis"])

    if not available_viz:
        st.warning("⚠️ No visualizable results found in the current run.")
        return

    viz_type = st.selectbox(
        "Visualization type",
        available_viz,
        help="Choose the type of visualization to display",
    )
    col1, col2 = st.columns([3, 1], gap="large")

    with col2:
        st.markdown("### 🎨 Display Options")
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
        st.markdown(f"### 📊 {viz_type}")
        if viz_type == "2D Network":
            fig = visualizer.plot_2d_network(
                st.session_state["processing_results"]["vertices"],
                st.session_state["processing_results"]["edges"],
                st.session_state["processing_results"]["network"],
                st.session_state["parameters"],
                color_by=color_scheme.lower().replace(" ", "_"),
                show_vertices=show_vertices,
                show_edges=show_edges,
                show_bifurcations=show_bifurcations,
            )
            st.plotly_chart(fig, use_container_width=True)
        elif viz_type == "3D Network":
            fig = visualizer.plot_3d_network(
                st.session_state["processing_results"]["vertices"],
                st.session_state["processing_results"]["edges"],
                st.session_state["processing_results"]["network"],
                st.session_state["parameters"],
                color_by=color_scheme.lower().replace(" ", "_"),
                show_vertices=show_vertices,
                show_edges=show_edges,
                show_bifurcations=show_bifurcations,
            )
            st.plotly_chart(fig, use_container_width=True)
        elif viz_type == "Depth Projection":
            fig = visualizer.plot_depth_statistics(
                st.session_state["processing_results"]["vertices"],
                st.session_state["processing_results"]["edges"],
                st.session_state["parameters"],
            )
            st.plotly_chart(fig, use_container_width=True)
        elif viz_type == "Strand Analysis":
            fig = visualizer.plot_strand_analysis(
                st.session_state["processing_results"]["network"],
                st.session_state["processing_results"]["vertices"],
                st.session_state["parameters"],
            )
            st.plotly_chart(fig, use_container_width=True)
        elif viz_type == "Energy Field":
            st.info(
                "Energy Field visualization is a 2D slice. Select slice axis and index in sidebar."
            )
            slice_axis = st.sidebar.selectbox(
                "Slice Axis", [0, 1, 2], format_func=lambda x: ["Y", "X", "Z"][x]
            )
            energy = st.session_state["processing_results"]["energy_data"]["energy"]
            slice_index = st.sidebar.number_input(
                "Slice Index", value=int(energy.shape[slice_axis] // 2)
            )
            fig = visualizer.plot_energy_field(
                st.session_state["processing_results"]["energy_data"],
                slice_axis=slice_axis,
                slice_index=slice_index,
            )
            st.plotly_chart(fig, use_container_width=True)

    if not web_app_facade._has_full_network_results(results):
        st.info("Complete the full network stage to unlock exports and the share report.")
        return

    st.markdown("### 💾 Export Options")
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    vertices = st.session_state["processing_results"]["vertices"]
    edges = st.session_state["processing_results"]["edges"]
    network = st.session_state["processing_results"]["network"]
    parameters = st.session_state["processing_results"]["parameters"]
    current_run_dir = st.session_state.get("current_run_dir")
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
    share_report_data = web_app_facade.generate_share_report_data(
        st.session_state["processing_results"],
        st.session_state.get("dataset_name", "SLAVV dataset"),
        st.session_state.get("image_shape", (100, 100, 50)),
    )
    web_app_facade._log_share_report_prepared_once(
        st.session_state.get("dataset_name", "SLAVV dataset"),
        share_report_data,
        st.session_state["processing_results"],
    )
    web_app_facade._update_run_task(
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
            from slavv.apps.share_report import record_share_event

            record_share_event(
                st.session_state,
                "share_report_downloaded",
                st.session_state.get("dataset_name", "SLAVV dataset"),
                share_report_data["signature"],
                extra={"report_file_name": share_report_data["file_name"]},
            )
            web_app_facade._update_run_task(
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
    share_metrics = st.session_state.get("share_report_metrics", {})
    st.caption(
        "Tracked share events this session: "
        f"requested={share_metrics.get('share_report_requested', 0)}, "
        f"downloaded={share_metrics.get('share_report_downloaded', 0)}"
    )
