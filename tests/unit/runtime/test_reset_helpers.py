"""Tests for stage-reset runtime helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slavv_python.runtime.models import StageSnapshot
from slavv_python.runtime.reset import (
    clear_stage_runtime_artifacts,
    remove_stage_dir_contents,
    reset_stage_snapshots,
)

if TYPE_CHECKING:
    from pathlib import Path


class _DummyController:
    def __init__(self, stage_dir: Path) -> None:
        self.stage_dir = stage_dir
        self.checkpoint_path = stage_dir.parent / "checkpoint_energy.pkl"
        self.manifest_path = stage_dir / "stage_manifest.json"
        self.state_path = stage_dir / "resume_state.json"


def test_remove_stage_dir_contents_removes_files_and_nested_directories(tmp_path):
    stage_dir = tmp_path / "stage"
    nested_dir = stage_dir / "nested" / "deeper"
    nested_dir.mkdir(parents=True)
    (stage_dir / "artifact.txt").write_text("artifact", encoding="utf-8")
    (nested_dir / "artifact.json").write_text("nested", encoding="utf-8")

    remove_stage_dir_contents(stage_dir)

    assert stage_dir.exists()
    assert list(stage_dir.iterdir()) == []


def test_clear_stage_runtime_artifacts_removes_checkpoint_manifest_and_state(tmp_path):
    stage_dir = tmp_path / "stages" / "energy"
    stage_dir.mkdir(parents=True)
    controller = _DummyController(stage_dir)
    controller.checkpoint_path.write_text("checkpoint", encoding="utf-8")
    controller.manifest_path.write_text("manifest", encoding="utf-8")
    controller.state_path.write_text("state", encoding="utf-8")
    (stage_dir / "artifact.txt").write_text("artifact", encoding="utf-8")

    clear_stage_runtime_artifacts(controller)

    assert not controller.checkpoint_path.exists()
    assert not controller.manifest_path.exists()
    assert not controller.state_path.exists()
    assert list(stage_dir.iterdir()) == []


def test_reset_stage_snapshots_reinitializes_requested_and_later_stages():
    stages = {
        "energy": StageSnapshot(name="energy", status="completed", progress=1.0),
        "vertices": StageSnapshot(name="vertices", status="running", progress=0.5),
        "edges": StageSnapshot(name="edges", status="completed", progress=1.0),
        "network": StageSnapshot(name="network", status="failed", progress=0.3),
    }

    affected = reset_stage_snapshots(stages, start_stage="vertices")

    assert affected == ["vertices", "edges", "network"]
    assert stages["energy"].status == "completed"
    assert stages["vertices"].status == "pending"
    assert stages["edges"].progress == 0.0
    assert stages["network"].detail == ""
