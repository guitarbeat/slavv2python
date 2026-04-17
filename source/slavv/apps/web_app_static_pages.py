"""Static content pages for the SLAVV Streamlit app."""

from __future__ import annotations

import streamlit as st


def show_home_page() -> None:
    """Display the home page with overview and quick start."""
    from slavv.apps import web_app as web_app_facade

    web_app_facade.show_dashboard_page()
    st.divider()
    st.markdown('<h2 class="section-header">Welcome to SLAVV</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1], gap="large", vertical_alignment="top")
    with col1:
        st.markdown(
            """
        ### 🔬 What is SLAVV?

        SLAVV (Segmentation-Less, Automated, Vascular Vectorization) is a sophisticated algorithm
        for extracting and analyzing vascular networks from 3D microscopy images. The method works
        through four main steps:

        1. **Energy Image Formation** - Multi-scale Hessian-based filtering to enhance vessel centerlines
        2. **Vertex Extraction** - Detection of vessel bifurcations and endpoints as local energy minima
        3. **Edge Extraction** - Tracing vessel segments between vertices through gradient descent
        4. **Network Construction** - Assembly of edges into connected vascular strands

        ### 🚀 Key Features

        - **Multi-scale Analysis** - Detects vessels across a wide range of sizes
        - **PSF Correction** - Accounts for microscope point spread function
        - **ML Curation** - Machine learning-assisted quality control
        - **Comprehensive Statistics** - Detailed network analysis and metrics
        - **Multiple Export Formats** - VMV, CASX, and custom formats
        - **Interactive Visualization** - 2D and 3D network rendering
        """
        )
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(
            """
        **✅ Ready to get started?**

        1. Navigate to **Image Processing** to upload and process your TIFF images
        2. Use **ML Curation** to refine vertex and edge detection
        3. Explore results in **Visualization** and **Analysis** pages
        """
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col2, st.container(height=400):
        st.markdown("### 📊 Quick Stats")
        st.metric("Supported Image Types", "TIFF", help="3D grayscale TIFF images")
        st.metric("Processing Steps", "4", help="Energy → Vertices → Edges → Network")
        st.metric("Export Formats", "5+", help="VMV, CASX, MAT, CSV, JSON")
        st.markdown("### 🔧 System Requirements")
        st.markdown(
            """
            - **Input**: 3D TIFF images
            - **Memory**: Depends on image size
            - **Processing**: Multi-threaded CPU
            - **Output**: Vector networks + statistics
            """
        )
        st.markdown("### 📚 Documentation")
        st.markdown(
            """
            - [Algorithm Overview](#)
            - [Parameter Guide](#)
            - [Export Formats](#)
            - [Troubleshooting](#)
            """
        )
        st.markdown("### 🎯 Workflow Control")
        st.markdown(
            """
            Like the original MATLAB scripts (`StartWorkflow`/`FinalWorkflow`), you can
            pause the pipeline early to inspect intermediate results or force the pipeline
            to recalculate specific steps to test parameter changes.
            """
        )


def show_about_page() -> None:
    """Display the about page with detailed information about SLAVV."""
    st.markdown('<h2 class="section-header">About SLAVV</h2>', unsafe_allow_html=True)
    st.markdown(
        """
    ### 🔬 Scientific Background

    SLAVV (Segmentation-Less, Automated, Vascular Vectorization) was developed to address the
    challenges of extracting vascular networks from large-scale microscopy volumes without
    requiring manual or error-prone segmentation steps.

    The algorithm uses a **multi-scale energy field** approach, where vessel centerlines are
    detected as local energy minima. This allows it to handle varying vessel diameters and
    low-contrast regions more robustly than threshold-based methods.

    ### 👨‍💻 Implementation Details

    This system is a modern Python implementation of the original SLAVV algorithm. Key improvements include:
    - **Performance**: Numba acceleration and multi-threaded processing.
    - **Scalability**: Chunk-based processing for large volumes.
    - **Modern UI**: Interactive Streamlit interface for parameter tuning and visualization.
    - **ML Integration**: Machine learning models for automatic quality control.

    ### 📝 Credits and License

    - **Original Algorithm**: Samuel Alexander Mihelic
    - **Python Port**: Developed for modern high-throughput analysis.
    - **License**: Provided as open-source for scientific research.

    For more information or to cite this work, please refer to the project documentation.
    """
    )
