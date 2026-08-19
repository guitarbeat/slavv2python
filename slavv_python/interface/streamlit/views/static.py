from __future__ import annotations

import streamlit as st

from .dashboard import show_dashboard_page


def show_home_page():
    """Display the home page with overview and quick start"""

    show_dashboard_page()
    st.divider()
    st.markdown('<h2 class="section-header">Welcome to SLAVV</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large", vertical_alignment="top")

    with col1:
        st.markdown("""
        ### What is SLAVV?

        SLAVV (Segmentation-Less, Automated, Vascular Vectorization) extracts a
        vascular Network from 3D microscopy volumes without a prior segmentation.
        The pipeline has four stages:

        1. **Energy** — multi-scale Hessian filtering that highlights vessel centerlines
        2. **Vertices** — local energy minima (seed points with position, radius, energy)
        3. **Edges** — **Tracing Discovery** on the public Paper Path, or **Watershed Discovery** on the Exact Route
        4. **Network** — strands and bifurcations assembled from the Edge Set

        ### Key Features

        - Multi-scale vessel sizes and optional PSF correction
        - Paper Path (`paper`) vs Exact Route (Watershed Discovery) on the same stages
        - Optional ML / automatic / desktop curation of vertices and edges
        - 2D and 3D visualization
        - In-app exports: VMV, CASX, CSV zip, and a shareable HTML report
        """)

        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("""
        **Ready to get started?**

        1. Navigate to **Image Processing** to upload and process your TIFF images
        2. Use **ML Curation** to refine vertex and edge detection
        3. Explore results in **Visualization** and **Analysis** pages
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2, st.container(height=400):
        st.markdown("### Quick Stats")

        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Supported Image Types", "TIFF", help="3D grayscale TIFF images")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Processing Steps", "4", help="Energy -> Vertices -> Edges -> Network")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "In-app exports",
            "4",
            help="Visualization downloads: VMV, CASX, CSV zip, and share HTML. The CLI can also write JSON.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### System Requirements")
        st.markdown("""
            - **Input**: 3D TIFF images
            - **Memory**: Depends on image size
            - **Processing**: Multi-threaded CPU
            - **Output**: Vector networks + statistics
            """)

        st.markdown("### Documentation")
        st.markdown("""
            Repo paths (open these files locally):

            - Tutorial: `docs/TUTORIAL.md`
            - Docs hub: `docs/README.md`
            - Two products (Paper Path vs Exact Route): `docs/reference/core/NEW_ENGINEER_START_HERE.md`
            """)

        st.markdown("### Workflow control")
        st.markdown("""
            On **Image Processing**, Pipeline Target stops after Energy, Vertices, Edges,
            or Network. Force Recalculation From ignores cached stage outputs from that
            stage onward.
            """)


def show_about_page():
    """Display the about page with detailed information about source."""
    st.markdown('<h2 class="section-header">About SLAVV</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### Scientific Background

    SLAVV (Segmentation-Less, Automated, Vascular Vectorization) was developed to address the
    challenges of extracting vascular networks from large-scale microscopy volumes without
    requiring manual or error-prone segmentation steps.

    The algorithm uses a **multi-scale energy field** approach, where vessel centerlines are
    detected as local energy minima. This allows it to handle varying vessel diameters and
    low-contrast regions more robustly than threshold-based methods.

    ### Implementation Details

    This app is the public **Paper Path** (`paper` profile): Tracing Discovery,
    the Streamlit workflow, and in-app visualization. The **Exact Route** is the
    MATLAB-faithful certification path (Watershed Discovery, `float64`, Fortran
    `[Y, X, Z]`) used by `slavv parity`, not this GUI's default.

    - Tutorial: `docs/TUTORIAL.md`
    - Two products: `docs/reference/core/NEW_ENGINEER_START_HERE.md`
    - Docs hub: `docs/README.md`

    ### Credits and License

    - **Method (publication):** Mihelic et al. 2021 (PLOS Computational Biology)
    - **MATLAB source of truth:** `external/Vectorization-Public/`
    - **This Python package:** SLAVV Paper Path + Exact Route in `slavv_python/`
    - **License:** GPL-3.0 (see `LICENSE`)
    """)
