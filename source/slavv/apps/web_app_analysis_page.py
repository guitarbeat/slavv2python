"""Analysis page for the SLAVV Streamlit app."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from slavv.analysis import MLCurator
from slavv.apps.curation_state import summarize_processing_counts
from slavv.apps.web_app_artifacts import _update_run_task
from slavv.visualization import NetworkVisualizer


def show_analysis_page() -> None:
    """Display the analysis page."""
    st.markdown('<h2 class="section-header">Network Analysis</h2>', unsafe_allow_html=True)
    if "processing_results" not in st.session_state:
        st.warning("⚠️ No processing results found. Please process an image first.")
        return

    results = st.session_state["processing_results"]
    if "network" not in results:
        st.warning(
            "⚠️ Analysis requires complete network extraction. Please run the pipeline up to the 'network' target."
        )
        return

    st.markdown(
        """
    Perform comprehensive statistical analysis on the vectorized vascular network. This section provides key metrics and detailed distributions.
    Corresponds to `SpecialOutput` parameters like `histograms`, `depth-stats`, `original-stats` in MATLAB.
    """
    )

    results = st.session_state["processing_results"]
    parameters = st.session_state["parameters"]
    _update_run_task(
        st.session_state.get("current_run_dir"),
        "analysis",
        status="completed",
        detail="Analysis dashboard viewed",
    )

    stats = st.session_state.get("analysis_stats")
    if stats is None:
        stats = summarize_processing_counts(results)

    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4, gap="small", vertical_alignment="center")
    with col1:
        st.metric("Total Length", f"{stats.get('total_length', 0):.1f} μm", help="Sum of all edge lengths")
    with col2:
        st.metric(
            "Volume Fraction",
            f"{stats.get('volume_fraction', 0):.3f}",
            help="Fraction of volume occupied by vessels",
        )
    with col3:
        st.metric(
            "Bifurcation Density",
            f"{stats.get('bifurcation_density', 0):.2f} /mm³",
            help="Bifurcations per cubic millimeter",
        )
    with col4:
        st.metric("Mean Radius", f"{stats.get('mean_radius', 0):.2f} μm", help="Average vessel radius")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🌳 Topology", "📏 Morphometry", "📊 Statistics"])
    visualizer = NetworkVisualizer()

    with tab1:
        st.markdown("#### Length and Radius Distributions")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.plotly_chart(
                visualizer.plot_strand_analysis(results["network"], results["vertices"], parameters),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(visualizer.plot_radius_distribution(results["vertices"]), use_container_width=True)
        st.markdown("#### Length-Weighted Histograms")
        st.caption(
            "Depth, radius, and inclination distributions weighted by segment length. Ported from `area_histogram_plotter.m`."
        )
        try:
            st.plotly_chart(
                visualizer.plot_length_weighted_histograms(
                    results.get("vertices", {}),
                    results.get("edges", {}),
                    results.get("parameters", {}),
                    number_of_bins=50,
                ),
                use_container_width=True,
            )
        except Exception as exc:
            st.info(f"Length-weighted histograms unavailable: {exc}")

    with tab2:
        st.markdown("#### Network Topology")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.plotly_chart(visualizer.plot_degree_distribution(results["network"]), use_container_width=True)
        with col2:
            connectivity_stats = pd.DataFrame(
                {
                    "Metric": [
                        "Connected Components",
                        "Average Path Length",
                        "Clustering Coefficient",
                        "Network Diameter",
                    ],
                    "Value": [
                        stats.get("num_connected_components", 0),
                        stats.get("avg_path_length", 0.0),
                        stats.get("clustering_coefficient", 0.0),
                        stats.get("network_diameter", 0.0),
                    ],
                }
            )
            st.dataframe(connectivity_stats, use_container_width=True)

    with tab3:
        st.markdown("#### Morphometric Analysis")
        st.plotly_chart(
            visualizer.plot_depth_statistics(results["vertices"], results["edges"], parameters),
            use_container_width=True,
        )
        col1, col2 = st.columns(2, gap="small")
        with col1:
            st.metric("Mean Tortuosity", f"{stats.get('mean_tortuosity', 0):.2f}", help="Average path tortuosity")
            st.metric("Tortuosity Std", f"{stats.get('tortuosity_std', 0):.2f}", help="Standard deviation of tortuosity")
        with col2:
            st.metric("Fractal Dimension", f"{stats.get('fractal_dimension', 0):.2f}", help="Complexity of network structure")
            st.metric("Lacunarity", f"{stats.get('lacunarity', 0):.2f}", help="Spatial heterogeneity of the network")

    with tab4:
        st.markdown("#### Complete Statistics Table")
        full_stats = pd.DataFrame(
            {
                "Metric": [
                    "Number of Strands",
                    "Number of Bifurcations",
                    "Number of Endpoints",
                    "Total Length (μm)",
                    "Mean Strand Length (μm)",
                    "Length Density (μm/μm³)",
                    "Volume Fraction",
                    "Mean Radius (μm)",
                    "Radius Std (μm)",
                    "Bifurcation Density (/mm³)",
                    "Surface Area (μm²)",
                    "Mean Tortuosity",
                    "Number of Connected Components",
                    "Average Path Length",
                    "Clustering Coefficient",
                    "Network Diameter",
                    "Fractal Dimension",
                    "Lacunarity",
                    "Tortuosity Std",
                ],
                "Value": [
                    stats.get("num_strands", 0),
                    stats.get("num_bifurcations", 0),
                    stats.get("num_endpoints", 0),
                    f"{stats.get('total_length', 0):.1f}",
                    f"{stats.get('mean_strand_length', 0):.1f}",
                    f"{stats.get('length_density', 0):.3f}",
                    f"{stats.get('volume_fraction', 0):.4f}",
                    f"{stats.get('mean_radius', 0):.2f}",
                    f"{stats.get('radius_std', 0):.2f}",
                    f"{stats.get('bifurcation_density', 0):.2f}",
                    f"{stats.get('surface_area', 0):.1f}",
                    f"{stats.get('mean_tortuosity', 0):.3f}",
                    stats.get("num_connected_components", 0),
                    f"{stats.get('avg_path_length', 0):.2f}",
                    f"{stats.get('clustering_coefficient', 0):.2f}",
                    f"{stats.get('network_diameter', 0):.2f}",
                    f"{stats.get('fractal_dimension', 0):.2f}",
                    f"{stats.get('lacunarity', 0):.2f}",
                    f"{stats.get('tortuosity_std', 0):.2f}",
                ],
            }
        )
        st.dataframe(
            full_stats,
            use_container_width=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", help="Statistic name"),
                "Value": st.column_config.TextColumn("Value", help="Computed value"),
            },
        )
        st.download_button(
            label="📥 Download Statistics CSV",
            data=full_stats.to_csv(index=False),
            file_name="network_statistics.csv",
            mime="text/csv",
        )
