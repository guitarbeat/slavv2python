"""Pytest configuration and fixtures.

Ensures the slavv package is importable when running tests from the repo root,
whether or not `pip install -e .` has been run.
"""
import sys
from pathlib import Path

# Add source/ to path so slavv is importable
repo_root = Path(__file__).resolve().parent.parent
source_dir = repo_root / "source"
if source_dir.exists() and str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))


def pytest_collection_modifyitems(items):
    """Auto-tag tests by folder so CI can select fast/full lanes."""
    for item in items:
        nodeid = item.nodeid.replace("\\", "/")
        if "/tests/unit/" in nodeid:
            item.add_marker("unit")
        elif "/tests/integration/" in nodeid:
            item.add_marker("integration")
        elif "/tests/ui/" in nodeid:
            item.add_marker("ui")
        elif "/tests/diagnostic/" in nodeid:
            item.add_marker("diagnostic")

        if "/tests/benchmarks/" in nodeid:
            item.add_marker("slow")
        if "regression" in nodeid:
            item.add_marker("regression")
