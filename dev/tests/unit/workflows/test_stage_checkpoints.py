from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from source.workflows.stage_checkpoints import (
    load_cached_stage_result,
    persist_stage_result,
    resolve_resumable_stage,
    stage_artifacts,
)

if TYPE_CHECKING:
    from pathlib import Path


class _DummyController:
    def __init__(self, stage_dir: Path) -> None:
        self.stage_dir = stage_dir
        self.checkpoint_path = stage_dir / "checkpoint.pkl"
        self.saved_payloads: list[dict[str, object]] = []
        self.completed_calls: list[dict[str, object]] = []
        self.loaded_payload = {"payload": "cached"}

    def load_checkpoint(self) -> dict[str, object]:
        return dict(self.loaded_payload)

    def save_checkpoint(self, payload: dict[str, object]) -> None:
        self.saved_payloads.append(dict(payload))

    def complete(self, **kwargs) -> None:
        self.completed_calls.append(kwargs)


def test_stage_artifacts_skips_resume_state_json(tmp_path):
    controller = _DummyController(tmp_path)
    (tmp_path / "resume_state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "payload.json").write_text("{}", encoding="utf-8")
    (tmp_path / "plots").mkdir()

    artifacts = stage_artifacts(controller)

    assert set(artifacts) == {"payload.json", "plots"}


def test_load_cached_stage_result_marks_stage_resumed(tmp_path):
    controller = _DummyController(tmp_path)
    controller.checkpoint_path.write_text("checkpoint", encoding="utf-8")
    (tmp_path / "payload.json").write_text("{}", encoding="utf-8")

    payload = load_cached_stage_result(controller, detail="Loaded checkpoint")

    assert payload == {"payload": "cached"}
    assert controller.completed_calls == [
        {
            "detail": "Loaded checkpoint",
            "artifacts": {
                "checkpoint.pkl": str(tmp_path / "checkpoint.pkl"),
                "payload.json": str(tmp_path / "payload.json"),
            },
            "resumed": True,
        }
    ]


def test_persist_stage_result_saves_and_marks_stage_complete(tmp_path):
    controller = _DummyController(tmp_path)
    (tmp_path / "summary.txt").write_text("done", encoding="utf-8")
    payload = {"payload": "new"}

    returned = persist_stage_result(controller, payload, detail="Stage complete")

    assert returned == payload
    assert controller.saved_payloads == [payload]
    assert controller.completed_calls == [
        {
            "detail": "Stage complete",
            "artifacts": {"summary.txt": str(tmp_path / "summary.txt")},
        }
    ]


def test_resolve_resumable_stage_loads_cached_payload_without_recomputing(tmp_path):
    controller = _DummyController(tmp_path)
    controller.loaded_payload = {"payload": "cached-result"}
    controller.checkpoint_path.write_text("checkpoint", encoding="utf-8")
    (tmp_path / "payload.json").write_text("{}", encoding="utf-8")
    compute_calls: list[str] = []

    payload = resolve_resumable_stage(
        controller,
        force_rerun=False,
        cached_log_label="Energy Field",
        cached_detail="Loaded checkpoint",
        success_detail="Computed checkpoint",
        compute_fn=lambda: compute_calls.append("computed"),
        logger=logging.getLogger("test-stage-checkpoints"),
    )

    assert payload == {"payload": "cached-result"}
    assert compute_calls == []


def test_resolve_resumable_stage_computes_and_persists_when_forced(tmp_path):
    controller = _DummyController(tmp_path)
    controller.checkpoint_path.write_text("checkpoint", encoding="utf-8")
    (tmp_path / "payload.json").write_text("{}", encoding="utf-8")

    payload = resolve_resumable_stage(
        controller,
        force_rerun=True,
        cached_log_label="Vertices",
        cached_detail="Loaded checkpoint",
        success_detail="Computed checkpoint",
        compute_fn=lambda: {"payload": "fresh-result"},
        logger=logging.getLogger("test-stage-checkpoints"),
    )

    assert payload == {"payload": "fresh-result"}
    assert controller.saved_payloads == [{"payload": "fresh-result"}]


