import pathlib
import sys
import warnings


def test_app_main_runs(tmp_path):
    """Smoke test that Streamlit app main executes without error."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / 'slavv-streamlit'))
    import app  # noqa: F401
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app.main()
