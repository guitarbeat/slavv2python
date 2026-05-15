from __future__ import annotations

import streamlit as st

from .pages.analysis import show_analysis_page
from .pages.curation import show_ml_curation_page
from .pages.processing import show_processing_page
from .pages.static import show_about_page, show_home_page
from .pages.visualization import show_visualization_page

PAGE_HANDLERS = {
    "🏠 Home": show_home_page,
    "⚙️ Image Processing": show_processing_page,
    "🤖 ML Curation": show_ml_curation_page,
    "📊 Visualization": show_visualization_page,
    "📈 Analysis": show_analysis_page,
    "Info: About": show_about_page,
}


def main():
    """Main Streamlit application shell."""
    st.markdown(
        '<h1 class="main-header">🩸 SLAVV - Vascular Vectorization System</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    **Segmentation-Less, Automated, Vascular Vectorization** - A comprehensive tool for analyzing
    vascular networks from grayscale, volumetric microscopy images.

    This Python/Streamlit implementation is based on the MATLAB SLAVV algorithm by Samuel Alexander Mihelic.
    The public workflow defaults to the native `paper` profile.
    """
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", list(PAGE_HANDLERS))
    PAGE_HANDLERS[page]()


__all__ = ["PAGE_HANDLERS", "main"]
