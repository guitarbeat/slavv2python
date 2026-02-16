import pytest
from unittest.mock import MagicMock
import importlib

# Skip if streamlit is not installed
st = pytest.importorskip("streamlit")

@pytest.fixture
def mock_streamlit_config(monkeypatch):
    """Prevent set_page_config from running and crashing tests."""
    monkeypatch.setattr(st, "set_page_config", lambda **kwargs: None)

def test_navigate_to_processing(mock_streamlit_config):
    """Test that navigate_to_processing updates session state."""
    from slavv.apps import web_app

    # Ensure clean state
    if "current_page" in st.session_state:
        del st.session_state["current_page"]

    web_app.navigate_to_processing()

    assert "current_page" in st.session_state
    assert st.session_state["current_page"] == "⚙️ Image Processing"

def test_render_empty_state(monkeypatch, mock_streamlit_config):
    """Test that render_empty_state calls appropriate streamlit functions."""
    markdown_mock = MagicMock()

    # Setup columns mock
    col_mock = MagicMock()
    col_mock.__enter__.return_value = col_mock
    col_mock.__exit__.return_value = None
    cols = [col_mock, col_mock, col_mock]
    columns_mock = MagicMock(return_value=cols)

    button_mock = MagicMock()

    monkeypatch.setattr(st, "markdown", markdown_mock)
    monkeypatch.setattr(st, "columns", columns_mock)
    monkeypatch.setattr(st, "button", button_mock)

    from slavv.apps import web_app
    # Reload to ensure we get fresh state if needed, though monkeypatch works on imported module
    importlib.reload(web_app)

    web_app.render_empty_state()

    # Verify markdown was called (for the message)
    assert markdown_mock.called
    args, _ = markdown_mock.call_args
    assert "No Results Yet" in args[0]

    # Verify columns were created
    columns_mock.assert_called_with([1, 2, 1])

    # Verify button was created with correct callback
    button_mock.assert_called()
    args, kwargs = button_mock.call_args
    assert "Go to Image Processing" in args[0]
    assert kwargs["on_click"] == web_app.navigate_to_processing
