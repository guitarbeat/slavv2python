"""Analysis page for the SLAVV Streamlit app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from slavv_python.interface.shared_services.exports import update_run_task
from slavv_python.interface.shared_state.analysis import (
    build_analysis_connectivity_rows,
    build_analysis_full_stats_rows,
    has_analysis_network,
    normalize_analysis_results,
    resolve_analysis_stats,
)
from slavv_python.interface.streamlit.empty_state import require_network
from slavv_python.visualization import NetworkVisualizer


def show_analysis_page() -> None:
    """Display the analysis page."""
    st.markdown('<h2 class="section-header">Network Analysis</h2>', unsafe_allow_html=True)
    raw = require_network()
    if raw is None:
        return

    results = normalize_analysis_results(raw)
    if not has_analysis_network(results):
        st.warning(
            "This step needs a complete Network. On Image Processing, set Pipeline Target "
            "to Full Pipeline (Network)."
        )
        return

    st.markdown(
        "Length, radius, topology, and morphometry for the current Network. "
        "Download the statistics table as CSV."
    )

    parameters = results["parameters"]
    update_run_task(
        st.session_state.get("current_run_dir"),
        "analysis",
        status="completed",
        detail="Analysis dashboard viewed",
    )

    stats = resolve_analysis_stats(results, st.session_state.get("analysis_stats"))

    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4, gap="small", vertical_alignment="center")
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Total Length", f"{stats.get('total_length', 0):.1f} um", help="Sum of all edge lengths"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Volume Fraction",
            f"{stats.get('volume_fraction', 0):.3f}",
            help="Fraction of volume occupied by vessels",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Bifurcation Density",
            f"{stats.get('bifurcation_density', 0):.2f} /mm^3",
            help="Bifurcations per cubic millimeter",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Mean Radius", f"{stats.get('mean_radius', 0):.2f} um", help="Average vessel radius"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Topology", "Morphometry", "Statistics"])
    visualizer = NetworkVisualizer()

    with tab1:
        st.markdown("#### Length and Radius Distributions")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.plotly_chart(
                visualizer.plot_strand_analysis(
                    results["network"], results["vertices"], parameters
                ),
                width="stretch",
            )
        with col2:
            st.plotly_chart(
                visualizer.plot_radius_distribution(results["vertices"]), width="stretch"
            )
        st.markdown("#### Length-Weighted Histograms")
        st.caption("Depth, radius, and inclination distributions weighted by segment length.")
        try:
            st.plotly_chart(
                visualizer.plot_length_weighted_histograms(
                    results.get("vertices", {}),
                    results.get("edges", {}),
                    results.get("parameters", {}),
                    number_of_bins=50,
                ),
                width="stretch",
            )
        except Exception as exc:
            st.info(f"Length-weighted histograms unavailable: {exc}")

    with tab2:
        st.markdown("#### Network Topology")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.plotly_chart(
                visualizer.plot_degree_distribution(results["network"]), width="stretch"
            )
        with col2:
            connectivity_stats = pd.DataFrame(build_analysis_connectivity_rows(stats))
            st.dataframe(connectivity_stats, width="stretch")

    with tab3:
        st.markdown("#### Morphometric Analysis")
        st.plotly_chart(
            visualizer.plot_depth_statistics(results["vertices"], results["edges"], parameters),
            width="stretch",
        )
        col1, col2 = st.columns(2, gap="small")
        with col1:
            st.metric(
                "Mean Tortuosity",
                f"{stats.get('mean_tortuosity', 0):.2f}",
                help="Average path tortuosity",
            )
            st.metric(
                "Tortuosity Std",
                f"{stats.get('tortuosity_std', 0):.2f}",
                help="Standard deviation of tortuosity",
            )
        with col2:
            st.metric(
                "Fractal Dimension",
                f"{stats.get('fractal_dimension', 0):.2f}",
                help="Complexity of network structure",
            )
            st.metric(
                "Lacunarity",
                f"{stats.get('lacunarity', 0):.2f}",
                help="Spatial heterogeneity of the network",
            )

    with tab4:
        st.markdown("#### Complete Statistics Table")
        full_stats = pd.DataFrame(build_analysis_full_stats_rows(stats))
        st.dataframe(
            full_stats,
            width="stretch",
            column_config={
                "Metric": st.column_config.TextColumn("Metric", help="Statistic name"),
                "Value": st.column_config.TextColumn("Value", help="Computed value"),
            },
        )
        st.download_button(
            label="Download Statistics CSV",
            data=full_stats.to_csv(index=False),
            file_name="network_statistics.csv",
            mime="text/csv",
        )
