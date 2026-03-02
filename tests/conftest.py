"""Pytest configuration and fixtures.

Ensures the slavv package is importable when running tests from the repo root,
whether or not `pip install -e .` has been run.
"""
import sys
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

# Add source/ to path so slavv is importable
repo_root = Path(__file__).resolve().parent.parent
source_dir = repo_root / "source"
if source_dir.exists() and str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))


def pytest_collection_modifyitems(items):
    """Auto-tag tests by folder so CI can select fast/full lanes."""
    for item in items:
        nodeid = item.nodeid.replace("\\", "/")
        if "tests/unit/" in nodeid:
            item.add_marker("unit")
        elif "tests/integration/" in nodeid:
            item.add_marker("integration")
        elif "tests/ui/" in nodeid:
            item.add_marker("ui")
        elif "tests/diagnostic/" in nodeid:
            item.add_marker("diagnostic")

        if "tests/benchmarks/" in nodeid:
            item.add_marker("slow")
        if "regression" in nodeid:
            item.add_marker("regression")


@pytest.fixture
def tmp_path():
    """Provide a writable temp directory without relying on pytest's lock-based tmpdir."""
    workspace_tmp_root = repo_root / "workspace" / "tmp_tests"
    workspace_tmp_root.mkdir(parents=True, exist_ok=True)
    path = workspace_tmp_root / f"run-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
