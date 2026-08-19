from __future__ import annotations

import streamlit as st

from .views.analysis import show_analysis_page
from .views.curation import show_ml_curation_page
from .views.processing import show_processing_page
from .views.static import show_about_page, show_home_page
from .views.visualization import show_visualization_page

PAGE_HANDLERS = {
    "Home": show_home_page,
    "Image Processing": show_processing_page,
    "Curation": show_ml_curation_page,
    "Visualization": show_visualization_page,
    "Analysis": show_analysis_page,
    "About": show_about_page,
}


def main():
    """Main Streamlit application shell."""
    st.markdown(
        '<h1 class="main-header">SLAVV — Vascular Vectorization</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    Extract a vascular **Network** from a 3D grayscale TIFF. The public default is the
    **Paper Path** (`paper` profile: Tracing Discovery). The **Exact Route**
    (Watershed Discovery, MATLAB-faithful) is available as an advanced edge method.
    """
    )

    st.sidebar.title("Pages")
    page = st.sidebar.selectbox("Choose a page:", list(PAGE_HANDLERS))
    PAGE_HANDLERS[page]()


__all__ = ["PAGE_HANDLERS", "main"]
