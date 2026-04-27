"""Tests for run-state layout helpers."""

from __future__ import annotations

import pytest
from source.runtime._run_state.layout import resolve_run_layout


def test_resolve_run_layout_structured_uses_staged_directories(tmp_path):
    layout = resolve_run_layout(
        run_dir=tmp_path / "run",
    )

    assert layout.run_root == tmp_path / "run"
    assert layout.metadata_dir == tmp_path / "run" / "99_Metadata"
    assert layout.artifacts_dir == tmp_path / "run" / "02_Output" / "python_results"
    assert layout.checkpoints_dir == layout.artifacts_dir / "checkpoints"
    assert layout.snapshot_path == layout.metadata_dir / "run_snapshot.json"


def test_resolve_run_layout_requires_run_dir():
    with pytest.raises(ValueError, match="run_dir is required for run state"):
        resolve_run_layout(run_dir=None)
