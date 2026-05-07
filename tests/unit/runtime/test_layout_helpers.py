"""Tests for run-state layout helpers."""

from __future__ import annotations

import pytest

from slavv_python.runtime.run_tracking.layout import resolve_run_layout


def test_resolve_run_layout_structured_uses_staged_directories(tmp_path):
    layout = resolve_run_layout(
        run_dir=tmp_path / "run",
    )

    assert layout.run_root == tmp_path / "run"
    assert layout.refs_dir == tmp_path / "run" / "00_Refs"
    assert layout.params_dir == tmp_path / "run" / "01_Params"
    assert layout.metadata_dir == tmp_path / "run" / "99_Metadata"
    assert layout.artifacts_dir == tmp_path / "run" / "02_Output" / "python_results"
    assert layout.analysis_dir == tmp_path / "run" / "03_Analysis"
    assert layout.checkpoints_dir == layout.artifacts_dir / "checkpoints"
    assert layout.normalized_dir == layout.analysis_dir / "normalized"
    assert layout.hashes_dir == layout.analysis_dir / "hashes"
    assert layout.snapshot_path == layout.metadata_dir / "run_snapshot.json"
    assert layout.manifest_path == layout.metadata_dir / "run_manifest.json"


def test_resolve_run_layout_requires_run_dir():
    with pytest.raises(ValueError, match="run_dir is required for run state"):
        resolve_run_layout(run_dir=None)
