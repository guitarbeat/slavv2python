"""Smoke the Streamlit entry file the same way ``slavv-app`` launches it."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

_ENTRY = Path(__file__).resolve().parents[2] / "slavv_python" / "interface" / "streamlit" / "app.py"
_STREAMLIT_PKG = Path(__file__).resolve().parents[2] / "slavv_python" / "interface" / "streamlit"


def test_streamlit_package_does_not_use_reserved_pages_folder() -> None:
    """Streamlit MPA treats a sibling ``pages/`` directory as extra scripts."""
    assert not (_STREAMLIT_PKG / "pages").exists()
    assert (_STREAMLIT_PKG / "views" / "processing.py").is_file()


def test_entry_script_loads_and_renders_shell() -> None:
    """``streamlit run app.py`` must import as a script and call ``main()``."""
    at = AppTest.from_file(str(_ENTRY), default_timeout=30)
    at.run()
    assert not at.exception
    nav = next(
        (
            box
            for box in list(at.selectbox) + list(at.sidebar.selectbox)
            if "Choose a page" in str(box.label)
        ),
        None,
    )
    assert nav is not None
    markdown = [el.value for el in at.markdown if isinstance(getattr(el, "value", None), str)]
    assert any("SLAVV" in value for value in markdown)
    for page in (
        "Home",
        "Image Processing",
        "Curation",
        "Visualization",
        "Analysis",
        "About",
    ):
        nav.select(page)
        at.run()
        assert not at.exception, page
