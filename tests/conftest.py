"""Pytest configuration and fixtures.

Ensures the slavv package is importable when running tests from the repo root,
whether or not `pip install -e .` has been run.
"""

from __future__ import annotations

import collections
import json
import shutil
import sys
from collections import abc as collections_abc
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

# Keep the legacy Python 3.7 scientific stack importable without flooding test
# output with third-party deprecations from old networkx/sklearn releases.
collections.Mapping = collections_abc.Mapping
collections.Set = collections_abc.Set
collections.Iterable = collections_abc.Iterable
np.int = int
np.float = float
np.bool = np.bool_

# Add repo root to path so `source` package imports are resolvable.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


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
    dev_tmp_root = repo_root / "workspace" / "tmp_tests"
    dev_tmp_root.mkdir(parents=True, exist_ok=True)
    path = dev_tmp_root / f"run-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json_fixture(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def comparison_metadata_builder():
    """Build reusable comparison metadata artifacts under ``99_Metadata``."""

    def _builder(
        run_dir: Path,
        *,
        run_snapshot: dict[str, object] | None = None,
        manifest_content: str | None = None,
        output_preflight: dict[str, object] | None = None,
    ) -> Path:
        metadata_dir = run_dir / "99_Metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        if run_snapshot is not None:
            _write_json_fixture(metadata_dir / "run_snapshot.json", run_snapshot)
        if output_preflight is not None:
            _write_json_fixture(metadata_dir / "output_preflight.json", output_preflight)
        if manifest_content is not None:
            (metadata_dir / "run_manifest.md").write_text(manifest_content, encoding="utf-8")
        return metadata_dir

    return _builder
