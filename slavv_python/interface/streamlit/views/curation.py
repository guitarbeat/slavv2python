"""Machine-learning curation page for the SLAVV Streamlit app."""

from __future__ import annotations

import os
import sys
from typing import cast

import pandas as pd
import plotly.express as px
import streamlit as st

from slavv_python.analytics import AutomaticCurator, MLCurator
from slavv_python.interface.streamlit.empty_state import require_edges
from slavv_python.interface.streamlit.navigation import switch_to
from slavv_python.interface.streamlit.services import curation as curation_services
from slavv_python.interface.streamlit.services.exports import update_run_task
from slavv_python.interface.streamlit.state.curation import (
    build_curation_stats_rows,
    summarize_processing_counts,
)
from slavv_python.interface.streamlit.views.manual_curation import (
    render_browser_manual_curation,
    render_manual_empty_state,
)


def _apply_curated_results(
    curated_vertices: dict[str, object],
    curated_edges: dict[str, object],
    *,
    curation_mode: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """Sync curated vertices and edges into session state with a rebuilt network."""
    return curation_services.apply_curated_results(
        st.session_state,
        curated_vertices,
        curated_edges,
        curation_mode=curation_mode,
    )


def desktop_curator_available() -> bool:
    """Return whether a desktop Qt/napari window can be opened from this process."""
    if os.environ.get("SLAVV_DISABLE_DESKTOP_CURATOR") == "1":
        return False
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    return True


def _run_interactive_curator(energy_data, vertices_data, edges_data, backend="qt"):
    """Import desktop curator backends lazily so the web app can load without GUI deps."""
    return curation_services.run_interactive_curator(
        energy_data,
        vertices_data,
        edges_data,
        backend=backend,
    )


def show_ml_curation_page():
    """Display manual, automatic, and model-based curation workflows."""
    st.header("Review and curate the network")

    if "processing_results" not in st.session_state:
        render_manual_empty_state()
        return

    results = require_edges()
    if results is None:
        render_manual_empty_state()
        return

    st.markdown(
        "Review the MATLAB-style projection workspace in two stages—vertices, then "
        "edges—and rebuild the shared network when the decisions are ready. The optional "
        "Python desktop viewer remains available for exploratory 3D review."
    )

    with st.sidebar:
        st.subheader("Curation")
        curation_type = st.radio(
            "Workflow",
            (
                "MATLAB-style browser curator",
                "Desktop manual review (MATLAB-style)",
                "Automatic filtering",
                "Model-assisted filtering",
            ),
            index=0,
            help="The MATLAB-style curator stays in this tab. The desktop viewer opens a separate Qt or napari window.",
        )

        if st.session_state.get("dataset_name"):
            st.caption(f"Dataset: {st.session_state['dataset_name']}")

    if curation_type == "MATLAB-style browser curator":
        render_browser_manual_curation(results)

    elif curation_type == "Desktop manual review (MATLAB-style)":
        st.markdown("#### Desktop 3D curator")
        st.info(
            "This Python desktop interface reproduces the four-panel MATLAB GCI layout: "
            "volume map, volume display, intensity histogram, and energy histogram. "
            "It opens in a **separate window**; close it to save and continue."
        )
        if not desktop_curator_available():
            st.warning(
                "The desktop interface is unavailable on this host. Use browser review or "
                "automatic filtering instead."
            )
        curator_backend_label = st.selectbox(
            "Interactive curator backend",
            ("Qt/PyVista (default)", "napari (experimental)"),
            help=(
                "Qt/PyVista preserves the current curator. napari is an experimental "
                "prototype with simpler image, point, and path editing."
            ),
        )
        curator_backend = "napari" if curator_backend_label.startswith("napari") else "qt"
        col1, col2 = st.columns(2)
        with col1:
            launch = st.button(
                "Launch desktop curator",
                type="primary",
                width=250,
                disabled=not desktop_curator_available(),
            )
            if launch:
                update_run_task(
                    st.session_state.get("current_run_dir"),
                    "manual_curation",
                    status="running",
                    detail="Interactive curator launched",
                )
                with st.status(
                    "Interactive Curator running in new window...", expanded=True
                ) as status:
                    st.warning(
                        "Check the taskbar for the new 3D window. Closing it saves and continues."
                    )
                    curated_vertices, curated_edges = _run_interactive_curator(
                        results["energy_data"],
                        results["vertices"],
                        results["edges"],
                        backend=curator_backend,
                    )
                    status.update(label="Rebuilding network after curation...", state="running")
                    try:
                        baseline_counts, current_counts = _apply_curated_results(
                            curated_vertices,
                            curated_edges,
                            curation_mode="Desktop curator (Qt / napari)",
                        )
                    except Exception as exc:
                        update_run_task(
                            st.session_state.get("current_run_dir"),
                            "manual_curation",
                            status="failed",
                            detail=f"Interactive curation could not rebuild the network: {exc!s}",
                        )
                        st.error(
                            "Curated vertices and edges were not applied because the network could not be rebuilt: "
                            f"{exc!s}"
                        )
                        st.stop()
                    update_run_task(
                        st.session_state.get("current_run_dir"),
                        "manual_curation",
                        status="completed",
                        detail="Interactive curation saved and network rebuilt",
                    )
                    status.update(label="Interactive Curation complete!", state="complete")
                    st.success("Desktop curator edits saved.")
                    st.caption(
                        "The downstream network, exports, and share report now use the curated vertices and edges."
                    )
                    c1, c2 = st.columns(2, gap="small")
                    with c1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            "Vertices",
                            current_counts["Vertices"],
                            delta=current_counts["Vertices"] - baseline_counts["Vertices"],
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                    with c2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            "Edges",
                            current_counts["Edges"],
                            delta=current_counts["Edges"] - baseline_counts["Edges"],
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

    elif curation_type == "Automatic filtering":
        st.markdown("#### Automatic filtering rules")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            vertex_energy_threshold = st.number_input(
                "Maximum vertex energy",
                min_value=-10.0,
                max_value=0.0,
                value=-0.1,
                step=0.01,
                help="Vertices with energy above this threshold will be removed.",
            )
            min_vertex_radius = st.number_input(
                "Minimum vertex radius (µm)",
                min_value=0.1,
                max_value=10.0,
                value=0.5,
                step=0.1,
                help="Vertices with radius below this will be removed.",
            )
        with col2:
            boundary_margin = st.number_input(
                "Image-boundary margin (voxels)",
                min_value=0,
                max_value=20,
                value=5,
                step=1,
                help="Vertices too close to image boundaries will be removed.",
            )
            contrast_threshold = st.number_input(
                "Minimum local contrast",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Vertices in low-contrast regions will be removed.",
            )
            min_edge_length = st.number_input(
                "Minimum edge length (µm)",
                min_value=0.1,
                max_value=20.0,
                value=2.0,
                step=0.1,
                help="Edges shorter than this will be removed.",
            )
            max_edge_tortuosity = st.number_input(
                "Maximum Edge Tortuosity",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.1,
                help="Edges with tortuosity above this will be removed.",
            )
            max_connection_distance = st.number_input(
                "Max Connection Distance (um)",
                min_value=0.1,
                max_value=10.0,
                value=5.0,
                step=0.1,
                help="Edges not properly connected to vertices within this distance will be removed.",
            )

        auto_curation_params = {
            "vertex_energy_threshold": vertex_energy_threshold,
            "min_vertex_radius": min_vertex_radius,
            "boundary_margin": boundary_margin,
            "contrast_threshold": contrast_threshold,
            "min_edge_length": min_edge_length,
            "max_edge_tortuosity": max_edge_tortuosity,
            "max_connection_distance": max_connection_distance,
            "image_shape": st.session_state["image_shape"],
        }

        if st.button("[Run] Start Automatic Curation", type="primary", width=250):
            update_run_task(
                st.session_state.get("current_run_dir"),
                "automatic_curation",
                status="running",
                detail="Automatic curation started",
            )
            with st.status("Performing automatic curation...", expanded=True) as status:
                curator = AutomaticCurator()
                curated_vertices = curator.curate_vertices_automatic(
                    results["vertices"], results["energy_data"], auto_curation_params
                )
                curated_edges = curator.curate_edges_automatic(
                    results["edges"], curated_vertices, auto_curation_params
                )
                status.update(label="Rebuilding network after curation...", state="running")
                try:
                    baseline_counts, current_counts = _apply_curated_results(
                        curated_vertices,
                        curated_edges,
                        curation_mode="Automatic filtering",
                    )
                except Exception as exc:
                    update_run_task(
                        st.session_state.get("current_run_dir"),
                        "automatic_curation",
                        status="failed",
                        detail=f"Automatic curation could not rebuild the network: {exc!s}",
                    )
                    st.error(
                        "Curated vertices and edges were not applied because the network could not be rebuilt: "
                        f"{exc!s}"
                    )
                    st.stop()
                update_run_task(
                    st.session_state.get("current_run_dir"),
                    "automatic_curation",
                    status="completed",
                    detail="Automatic curation complete and network rebuilt",
                )
                st.success("Automatic curation complete.")
                status.update(label="Automatic curation complete!", state="complete")
                st.caption(
                    "The downstream network, exports, and share report now use the curated vertices and edges."
                )
                col1, col2 = st.columns(2, gap="small")
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Vertices",
                        current_counts["Vertices"],
                        delta=current_counts["Vertices"] - baseline_counts["Vertices"],
                        help="Change relative to the pre-curation baseline",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Edges",
                        current_counts["Edges"],
                        delta=current_counts["Edges"] - baseline_counts["Edges"],
                        help="Change relative to the pre-curation baseline",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

    elif curation_type == "Model-assisted filtering":
        st.markdown("#### Machine Learning Curation Parameters")
        st.info("Upload pre-trained models or provide CSV training data to train new classifiers.")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.selectbox(
                "Vertex curation method",
                ["machine-auto"],
                help="Choose how to curate detected vertices. Corresponds to `VertexCuration` parameter in MATLAB.",
            )
            vertex_model_file = st.file_uploader(
                "Vertex model (.joblib)",
                type=["joblib", "pkl"],
                help="Upload a pre-trained vertex classifier",
            )
            vertex_training_data = st.file_uploader(
                "Vertex training data (.csv)",
                type=["csv"],
                help="CSV with vertex features and a 'label' column",
            )
            vertex_confidence_threshold = st.slider(
                "Vertex Confidence threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Minimum confidence score for keeping vertices",
            )
        with col2:
            st.selectbox(
                "Edge curation method",
                ["machine-auto"],
                help="Choose how to curate detected edges. Corresponds to `EdgeCuration` parameter in MATLAB.",
            )
            edge_model_file = st.file_uploader(
                "Edge model (.joblib)",
                type=["joblib", "pkl"],
                help="Upload a pre-trained edge classifier",
            )
            edge_training_data = st.file_uploader(
                "Edge training data (.csv)",
                type=["csv"],
                help="CSV with edge features and a 'label' column",
            )
            edge_confidence_threshold = st.slider(
                "Edge Confidence threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Minimum confidence score for keeping edges",
            )

        if st.button("[Train] Train Models", type="secondary", width=250):
            if vertex_training_data is None and edge_training_data is None:
                st.error("Please upload training data for vertices, edges, or both.")
            else:
                update_run_task(
                    st.session_state.get("current_run_dir"),
                    "ml_training",
                    status="running",
                    detail="Training ML models",
                )
                with st.status("Training ML models...", expanded=True) as status:
                    ml_curator = MLCurator()
                    if vertex_training_data is not None:
                        df_v = pd.read_csv(vertex_training_data)
                        X_v = df_v.drop(columns=["label"]).values
                        y_v = df_v["label"].values
                        res_v = ml_curator.train_vertex_classifier(X_v, y_v)
                        st.write(f"Vertex test accuracy: {res_v['test_accuracy']:.3f}")
                    if edge_training_data is not None:
                        df_e = pd.read_csv(edge_training_data)
                        X_e = df_e.drop(columns=["label"]).values
                        y_e = df_e["label"].values
                        res_e = ml_curator.train_edge_classifier(X_e, y_e)
                        st.write(f"Edge test accuracy: {res_e['test_accuracy']:.3f}")
                    st.session_state["ml_curator"] = ml_curator
                    update_run_task(
                        st.session_state.get("current_run_dir"),
                        "ml_training",
                        status="completed",
                        detail="ML models trained",
                    )
                    status.update(label="Training complete!", state="complete")
                    st.success("Models trained.")

        if st.button("Start ML curation", type="primary", width=250):
            update_run_task(
                st.session_state.get("current_run_dir"),
                "ml_curation",
                status="running",
                detail="ML curation started",
            )
            with st.status("Performing ML curation...", expanded=True) as status:
                ml_curator = cast("MLCurator | None", st.session_state.get("ml_curator"))
                if ml_curator is None:
                    ml_curator = MLCurator()
                    ml_curator.load_models(vertex_model_file, edge_model_file)
                if ml_curator.vertex_classifier is None and ml_curator.edge_classifier is None:
                    st.error("ML models are not loaded or trained, so ML curation cannot run.")
                    update_run_task(
                        st.session_state.get("current_run_dir"),
                        "ml_curation",
                        status="failed",
                        detail="ML models were not available for curation",
                    )
                    st.stop()

                curated_vertices = results["vertices"]
                if ml_curator.vertex_classifier is not None:
                    curated_vertices = ml_curator.curate_vertices(
                        results["vertices"],
                        results["energy_data"],
                        st.session_state["image_shape"],
                        vertex_confidence_threshold,
                    )

                curated_edges = results["edges"]
                if ml_curator.edge_classifier is not None:
                    curated_edges = ml_curator.curate_edges(
                        results["edges"],
                        curated_vertices,
                        results["energy_data"],
                        edge_confidence_threshold,
                    )
                status.update(label="Rebuilding network after curation...", state="running")
                try:
                    baseline_counts, current_counts = _apply_curated_results(
                        curated_vertices,
                        curated_edges,
                        curation_mode="Model-assisted filtering",
                    )
                except Exception as exc:
                    update_run_task(
                        st.session_state.get("current_run_dir"),
                        "ml_curation",
                        status="failed",
                        detail=f"ML curation could not rebuild the network: {exc!s}",
                    )
                    st.error(
                        "Curated vertices and edges were not applied because the network could not be rebuilt: "
                        f"{exc!s}"
                    )
                    st.stop()
                st.success("ML curation complete.")
                status.update(label="ML curation complete!", state="complete")
                update_run_task(
                    st.session_state.get("current_run_dir"),
                    "ml_curation",
                    status="completed",
                    detail="ML curation complete and network rebuilt",
                )
                st.caption(
                    "The downstream network, exports, and share report now use the curated vertices and edges."
                )
                col1, col2 = st.columns(2, gap="small")
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Vertices",
                        current_counts["Vertices"],
                        delta=current_counts["Vertices"] - baseline_counts["Vertices"],
                        help="Change relative to the pre-curation baseline",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Edges",
                        current_counts["Edges"],
                        delta=current_counts["Edges"] - baseline_counts["Edges"],
                        help="Change relative to the pre-curation baseline",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("[Stats] Show Curation Statistics", width=250):
        st.markdown("### [Graph] Curation Results")
        baseline_counts = st.session_state.get("curation_baseline_counts")
        if baseline_counts is None:
            st.info(
                "No curation has been applied yet. Run a curation step to compare before/after counts."
            )
        else:
            current_counts = summarize_processing_counts(st.session_state["processing_results"])
            curation_stats = pd.DataFrame(
                build_curation_stats_rows(baseline_counts, current_counts)
            )
            curation_mode = st.session_state.get("last_curation_mode")
            if curation_mode:
                st.caption(
                    f"Most recent curation mode: {curation_mode}. The network was rebuilt after the curated vertices and edges were applied."
                )
            st.dataframe(curation_stats, width="stretch")
            fig = px.bar(
                curation_stats,
                x="Component",
                y=["Original", "Current"],
                title="Curation Results",
                barmode="group",
            )
            st.plotly_chart(fig, width="stretch")

    if st.session_state.get("last_curation_mode"):
        st.divider()
        next_columns = st.columns(2, gap="small")
        if next_columns[0].button(
            "Inspect curated network",
            type="primary",
            icon=":material/view_in_ar:",
            width="stretch",
        ):
            switch_to("visualization")
        if next_columns[1].button(
            "Open analysis",
            icon=":material/analytics:",
            width="stretch",
        ):
            switch_to("analysis")
