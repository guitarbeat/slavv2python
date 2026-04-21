from __future__ import annotations

import logging

import pytest

from slavv.workflows.pipeline_stages import (
    resolve_edges_stage,
    resolve_energy_stage,
    resolve_network_stage,
    resolve_stage_with_checkpoint,
    resolve_vertices_stage,
)


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


def test_resolve_energy_stage_uses_standard_stage_metadata(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "slavv.workflows.pipeline_stages.resolve_stage_with_checkpoint",
        lambda **kwargs: captured.update(kwargs) or {"stage": "energy"},
    )

    result = resolve_energy_stage(
        run_context="context",
        force_rerun=True,
        fallback_fn=lambda: {"source": "fallback"},
        resumable_fn=lambda controller: {"source": "resumable"},
        logger=logging.getLogger("test-pipeline-stages"),
    )

    assert result == {"stage": "energy"}
    assert captured["stage_name"] == "energy"
    assert captured["cached_log_label"] == "Energy Field"
    assert captured["cached_detail"] == "Loaded energy checkpoint"
    assert captured["success_detail"] == "Energy field ready"


def test_resolve_vertices_stage_uses_standard_stage_metadata(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "slavv.workflows.pipeline_stages.resolve_stage_with_checkpoint",
        lambda **kwargs: captured.update(kwargs) or {"stage": "vertices"},
    )

    result = resolve_vertices_stage(
        run_context="context",
        force_rerun=False,
        fallback_fn=lambda: {"source": "fallback"},
        resumable_fn=lambda controller: {"source": "resumable"},
        logger=logging.getLogger("test-pipeline-stages"),
    )

    assert result == {"stage": "vertices"}
    assert captured["stage_name"] == "vertices"
    assert captured["cached_log_label"] == "Vertices"


def test_resolve_edges_stage_selects_watershed_callbacks(monkeypatch):
    captured: dict[str, object] = {}
    calls: list[str] = []

    monkeypatch.setattr(
        "slavv.workflows.pipeline_stages.resolve_stage_with_checkpoint",
        lambda **kwargs: captured.update(kwargs)
        or kwargs["fallback_fn"]()
        or kwargs["compute_fn"]("controller"),
    )

    resolve_edges_stage(
        run_context="context",
        force_rerun=True,
        edge_method="watershed",
        tracing_fallback_fn=lambda: calls.append("tracing-fallback") or {"mode": "tracing"},
        watershed_fallback_fn=lambda: calls.append("watershed-fallback") or {"mode": "watershed"},
        tracing_resumable_fn=lambda controller: calls.append("tracing-resumable")
        or {"mode": "tracing"},
        watershed_resumable_fn=lambda controller: calls.append("watershed-resumable")
        or {"mode": "watershed"},
        logger=logging.getLogger("test-pipeline-stages"),
    )

    assert captured["stage_name"] == "edges"
    assert captured["cached_log_label"] == "Edges"
    assert calls == ["watershed-fallback"]


def test_resolve_network_stage_uses_standard_stage_metadata(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "slavv.workflows.pipeline_stages.resolve_stage_with_checkpoint",
        lambda **kwargs: captured.update(kwargs) or {"stage": "network"},
    )

    result = resolve_network_stage(
        run_context="context",
        force_rerun=False,
        fallback_fn=lambda: {"source": "fallback"},
        resumable_fn=lambda controller: {"source": "resumable"},
        logger=logging.getLogger("test-pipeline-stages"),
    )

    assert result == {"stage": "network"}
    assert captured["stage_name"] == "network"
    assert captured["cached_log_label"] == "Network"
