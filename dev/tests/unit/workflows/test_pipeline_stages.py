from __future__ import annotations

import logging

import pytest

from slavv.workflows.pipeline_stages import resolve_stage_with_checkpoint


class _DummyRunContext:
    def __init__(self, controller) -> None:
        self.controller = controller
        self.failed: list[tuple[str, str]] = []
        self.stage_calls: list[str] = []

    def stage(self, stage_name: str):
        self.stage_calls.append(stage_name)
        return self.controller

    def fail_stage(self, stage_name: str, exc: Exception) -> None:
        self.failed.append((stage_name, str(exc)))


class _DummyController:
    def __init__(self) -> None:
        self.checkpoint_path = type("_PathLike", (), {"exists": lambda self: False})()


def test_resolve_stage_with_checkpoint_uses_fallback_without_run_context():
    result = resolve_stage_with_checkpoint(
        run_context=None,
        force_rerun=False,
        stage_name="energy",
        cached_log_label="Energy",
        cached_detail="Loaded",
        success_detail="Done",
        fallback_fn=lambda: {"source": "fallback"},
        compute_fn=lambda: {"source": "compute"},
        logger=logging.getLogger("test-pipeline-stages"),
    )

    assert result == {"source": "fallback"}


def test_resolve_stage_with_checkpoint_uses_resumable_helper(monkeypatch):
    controller = _DummyController()
    run_context = _DummyRunContext(controller)
    seen: list[tuple[object, bool, str]] = []
    compute_calls: list[object] = []

    monkeypatch.setattr(
        "slavv.workflows.pipeline_stages.resolve_resumable_stage",
        lambda stage_controller, *, force_rerun, cached_log_label, compute_fn, **kwargs: (
            seen.append((stage_controller, force_rerun, cached_log_label))
            or compute_fn()
        ),
    )

    result = resolve_stage_with_checkpoint(
        run_context=run_context,
        force_rerun=True,
        stage_name="edges",
        cached_log_label="Edges",
        cached_detail="Loaded",
        success_detail="Done",
        fallback_fn=lambda: {"source": "fallback"},
        compute_fn=lambda stage_controller: (
            compute_calls.append(stage_controller) or {"source": "compute"}
        ),
        logger=logging.getLogger("test-pipeline-stages"),
    )

    assert result == {"source": "compute"}
    assert run_context.stage_calls == ["edges"]
    assert seen == [(controller, True, "Edges")]
    assert compute_calls == [controller]


def test_resolve_stage_with_checkpoint_marks_stage_failed_before_reraising(monkeypatch):
    controller = _DummyController()
    run_context = _DummyRunContext(controller)

    def boom(*args, **kwargs):
        raise RuntimeError("stage blew up")

    monkeypatch.setattr("slavv.workflows.pipeline_stages.resolve_resumable_stage", boom)

    with pytest.raises(RuntimeError, match="stage blew up"):
        resolve_stage_with_checkpoint(
            run_context=run_context,
            force_rerun=False,
            stage_name="network",
            cached_log_label="Network",
            cached_detail="Loaded",
            success_detail="Done",
            fallback_fn=lambda: {"source": "fallback"},
            compute_fn=lambda: {"source": "compute"},
            logger=logging.getLogger("test-pipeline-stages"),
        )

    assert run_context.failed == [("network", "stage blew up")]
