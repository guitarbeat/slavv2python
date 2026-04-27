from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from source.workflows.pipeline_setup import (
    PreparedPipelineRun,
    create_run_context,
    effective_run_dir,
    emit_progress,
    force_rerun_flags,
    initialize_run_context,
    prepare_pipeline_run,
    preprocess_image,
    validate_stage_control,
)


class _DummyRunContext:
    def __init__(self) -> None:
        self.run_root = Path("run-root")
        self.metadata_dir = Path("run-root") / "99_Metadata"
        self.resume_calls: list[dict[str, object]] = []
        self.reset_calls: list[str] = []
        self.status_calls: list[dict[str, object]] = []
        self.preprocess_marked = False
        self.failed: list[tuple[str, str]] = []

    def ensure_resume_allowed(self, **kwargs) -> None:
        self.resume_calls.append(kwargs)

    def reset_pipeline_state_from(self, stage_name: str) -> None:
        self.reset_calls.append(stage_name)

    def mark_run_status(self, status: str, **kwargs) -> None:
        self.status_calls.append({"status": status, **kwargs})

    def mark_preprocess_complete(self) -> None:
        self.preprocess_marked = True

    def fail_stage(self, stage_name: str, exc: Exception) -> None:
        self.failed.append((stage_name, str(exc)))


def test_validate_stage_control_rejects_unknown_stage():
    with pytest.raises(ValueError, match="stop_after must be one of"):
        validate_stage_control("bogus", "stop_after")


def test_emit_progress_invokes_callback():
    seen: list[tuple[float, str]] = []

    emit_progress(lambda fraction, stage: seen.append((fraction, stage)), 0.4, "energy")

    assert seen == [(0.4, "energy")]


def test_effective_run_dir_creates_temp_dir_when_event_callback_is_present():
    run_dir = effective_run_dir(None, lambda _event: None)

    assert run_dir is not None
    assert Path(run_dir).name.startswith("slavv_run_")


def test_create_run_context_uses_factory_and_pipeline_provenance():
    calls: list[dict[str, object]] = []

    def factory(**kwargs):
        calls.append(kwargs)
        return "context"

    context = create_run_context(
        "resolved-run-dir",
        "input-fingerprint",
        "params-fingerprint",
        np.zeros((2, 3, 4), dtype=np.float32),
        "edges",
        None,
        run_context_factory=factory,
    )

    assert context == "context"
    assert calls[0]["target_stage"] == "edges"
    assert calls[0]["provenance"] == {
        "source": "pipeline",
        "image_shape": [2, 3, 4],
        "stop_after": "edges",
    }


def test_initialize_run_context_writes_params_and_marks_running(monkeypatch):
    writes: list[tuple[Path, dict[str, object]]] = []
    run_context = _DummyRunContext()

    monkeypatch.setattr(
        "source.workflows.pipeline_setup.atomic_write_json",
        lambda path, payload: writes.append((path, payload)),
    )

    initialize_run_context(
        run_context,
        input_fingerprint="input",
        params_fingerprint="params",
        force_rerun_from="vertices",
        parameters={"alpha": 1},
    )

    assert run_context.resume_calls == [
        {
            "input_fingerprint": "input",
            "params_fingerprint": "params",
            "force_rerun_from": "vertices",
        }
    ]
    assert run_context.reset_calls == ["vertices"]
    assert writes == [(Path("run-root") / "99_Metadata" / "validated_params.json", {"alpha": 1})]
    assert run_context.status_calls[0]["status"] == "running"
    assert run_context.status_calls[0]["current_stage"] == "preprocess"


def test_force_rerun_flags_marks_requested_stage_and_later_stages():
    flags = force_rerun_flags("vertices")

    assert flags == {
        "energy": False,
        "vertices": True,
        "edges": True,
        "network": True,
    }


def test_preprocess_image_marks_completion(monkeypatch):
    run_context = _DummyRunContext()
    image = np.ones((2, 2, 2), dtype=np.float32)

    monkeypatch.setattr(
        "source.workflows.pipeline_setup.utils.preprocess_image",
        lambda image_arg, parameters_arg: image_arg * 2,
    )

    result = preprocess_image(image, {}, run_context)

    assert np.array_equal(result, image * 2)
    assert run_context.preprocess_marked is True


def test_preprocess_image_marks_failure_before_reraising(monkeypatch):
    run_context = _DummyRunContext()

    def boom(_image, _parameters):
        raise RuntimeError("preprocess broke")

    monkeypatch.setattr("source.workflows.pipeline_setup.utils.preprocess_image", boom)

    with pytest.raises(RuntimeError, match="preprocess broke"):
        preprocess_image(np.ones((2, 2, 2), dtype=np.float32), {}, run_context)

    assert run_context.failed == [("preprocess", "preprocess broke")]


def test_prepare_pipeline_run_validates_inputs_and_initializes_context(monkeypatch):
    run_context = _DummyRunContext()

    monkeypatch.setattr(
        "source.workflows.pipeline_setup.utils.validate_parameters",
        lambda params: {"validated": True, **params},
    )
    monkeypatch.setattr(
        "source.workflows.pipeline_setup.fingerprint_array",
        lambda image: "image-fingerprint",
    )
    monkeypatch.setattr(
        "source.workflows.pipeline_setup.fingerprint_jsonable",
        lambda params: "params-fingerprint",
    )
    monkeypatch.setattr(
        "source.workflows.pipeline_setup.effective_run_dir",
        lambda run_dir, event_callback: "resolved-run-dir",
    )

    create_calls: list[dict[str, object]] = []

    def fake_create_run_context(
        effective_run_dir_value,
        input_fingerprint,
        params_fingerprint,
        image,
        stop_after,
        event_callback,
        *,
        run_context_factory,
    ):
        create_calls.append(
            {
                "effective_run_dir_value": effective_run_dir_value,
                "input_fingerprint": input_fingerprint,
                "params_fingerprint": params_fingerprint,
                "shape": tuple(image.shape),
                "stop_after": stop_after,
                "run_context_factory": run_context_factory,
            }
        )
        return run_context

    monkeypatch.setattr(
        "source.workflows.pipeline_setup.create_run_context", fake_create_run_context
    )

    prepared = prepare_pipeline_run(
        np.ones((2, 3, 4), dtype=np.float32),
        {"alpha": 1},
        run_dir=None,
        stop_after="edges",
        force_rerun_from="vertices",
        event_callback=None,
        run_context_factory=_DummyRunContext,
    )

    assert isinstance(prepared, PreparedPipelineRun)
    assert prepared.parameters == {"validated": True, "alpha": 1}
    assert prepared.run_context is run_context
    assert prepared.force_rerun == {
        "energy": False,
        "vertices": True,
        "edges": True,
        "network": True,
    }
    assert create_calls == [
        {
            "effective_run_dir_value": "resolved-run-dir",
            "input_fingerprint": "image-fingerprint",
            "params_fingerprint": "params-fingerprint",
            "shape": (2, 3, 4),
            "stop_after": "edges",
            "run_context_factory": _DummyRunContext,
        }
    ]
    assert run_context.resume_calls == [
        {
            "input_fingerprint": "image-fingerprint",
            "params_fingerprint": "params-fingerprint",
            "force_rerun_from": "vertices",
        }
    ]
