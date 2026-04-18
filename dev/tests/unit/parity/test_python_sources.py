from __future__ import annotations

import json
from typing import TYPE_CHECKING

import joblib

from slavv.parity._comparison.python_sources import _load_python_results_from_checkpoints

if TYPE_CHECKING:
    from pathlib import Path


def test_load_python_results_from_checkpoints_ignores_missing_candidate_lifecycle(tmp_path: Path):
    python_root = tmp_path / "python_results"
    checkpoint_dir = python_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    joblib.dump({"positions": [[1, 2, 3]]}, checkpoint_dir / "checkpoint_vertices.pkl")
    joblib.dump({"traces": []}, checkpoint_dir / "checkpoint_edges.pkl")
    joblib.dump({"strands": []}, checkpoint_dir / "checkpoint_network.pkl")

    result = _load_python_results_from_checkpoints(python_root)

    assert result is not None
    assert "candidate_lifecycle" not in result


def test_load_python_results_from_checkpoints_ignores_invalid_candidate_lifecycle(tmp_path: Path):
    python_root = tmp_path / "python_results"
    checkpoint_dir = python_root / "checkpoints"
    stage_dir = python_root / "stages" / "edges"
    checkpoint_dir.mkdir(parents=True)
    stage_dir.mkdir(parents=True)

    joblib.dump({"positions": [[1, 2, 3]]}, checkpoint_dir / "checkpoint_vertices.pkl")
    joblib.dump({"traces": []}, checkpoint_dir / "checkpoint_edges.pkl")
    joblib.dump({"strands": []}, checkpoint_dir / "checkpoint_network.pkl")
    (stage_dir / "candidate_lifecycle.json").write_text("{not-json", encoding="utf-8")

    result = _load_python_results_from_checkpoints(python_root)

    assert result is not None
    assert "candidate_lifecycle" not in result


def test_load_python_results_from_checkpoints_includes_valid_candidate_lifecycle(tmp_path: Path):
    python_root = tmp_path / "python_results"
    checkpoint_dir = python_root / "checkpoints"
    stage_dir = python_root / "stages" / "edges"
    checkpoint_dir.mkdir(parents=True)
    stage_dir.mkdir(parents=True)

    joblib.dump({"positions": [[1, 2, 3]]}, checkpoint_dir / "checkpoint_vertices.pkl")
    joblib.dump({"traces": []}, checkpoint_dir / "checkpoint_edges.pkl")
    joblib.dump({"strands": []}, checkpoint_dir / "checkpoint_network.pkl")
    (stage_dir / "candidate_lifecycle.json").write_text(
        json.dumps({"events": [{"resolution_reason": "accepted_seed_origin"}]}),
        encoding="utf-8",
    )

    result = _load_python_results_from_checkpoints(python_root)

    assert result is not None
    assert result["candidate_lifecycle"]["events"][0]["resolution_reason"] == "accepted_seed_origin"
