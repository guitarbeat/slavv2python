from __future__ import annotations

from source.workflows.pipeline_runner import (
    PipelineStageStep,
    advance_pipeline_stage,
    build_standard_pipeline_steps,
    run_pipeline_stage_sequence,
)


class _DummyRunContext:
    def __init__(self) -> None:
        self.stop_after_calls: list[str | None] = []

    def finalize_run(self, *, stop_after: str | None = None) -> None:
        self.stop_after_calls.append(stop_after)


def test_advance_pipeline_stage_records_payload_and_emits_progress():
    results: dict[str, object] = {"parameters": {}}
    progress: list[tuple[float, str]] = []

    early = advance_pipeline_stage(
        results,
        result_key="vertices",
        payload={"positions": []},
        stage_name="vertices",
        progress_fraction=0.6,
        progress_callback=lambda fraction, stage: progress.append((fraction, stage)),
        stop_after=None,
        run_context=None,
    )

    assert early is None
    assert results["vertices"] == {"positions": []}
    assert progress == [(0.6, "vertices")]


def test_advance_pipeline_stage_returns_early_when_stop_stage_matches():
    results: dict[str, object] = {"parameters": {}}
    run_context = _DummyRunContext()

    early = advance_pipeline_stage(
        results,
        result_key="network",
        payload={"strands": []},
        stage_name="network",
        progress_fraction=1.0,
        progress_callback=None,
        stop_after="network",
        run_context=run_context,
    )

    assert early is not None
    assert early["network"]["strands"] == []
    assert early["network"]["bifurcations"].shape == (0,)
    assert run_context.stop_after_calls == ["network"]


def test_run_pipeline_stage_sequence_executes_steps_in_order():
    results: dict[str, object] = {"parameters": {}}
    calls: list[str] = []

    early = run_pipeline_stage_sequence(
        results,
        steps=[
            PipelineStageStep(
                result_key="energy_data",
                stage_name="energy",
                progress_fraction=0.4,
                resolve_fn=lambda: calls.append("energy") or {"energy": []},
            ),
            PipelineStageStep(
                result_key="vertices",
                stage_name="vertices",
                progress_fraction=0.6,
                resolve_fn=lambda: calls.append("vertices") or {"positions": []},
            ),
        ],
        progress_callback=None,
        stop_after=None,
        run_context=None,
    )

    assert early is None
    assert calls == ["energy", "vertices"]
    assert results["energy_data"] == {"energy": []}
    assert results["vertices"] == {"positions": []}


def test_build_standard_pipeline_steps_uses_canonical_stage_order():
    calls: list[str] = []

    steps = build_standard_pipeline_steps(
        resolve_energy=lambda: calls.append("energy") or {"energy": []},
        resolve_vertices=lambda: calls.append("vertices") or {"positions": []},
        resolve_edges=lambda: calls.append("edges") or {"traces": []},
        resolve_network=lambda: calls.append("network") or {"strands": []},
    )

    assert [step.result_key for step in steps] == [
        "energy_data",
        "vertices",
        "edges",
        "network",
    ]
    assert [step.stage_name for step in steps] == [
        "energy",
        "vertices",
        "edges",
        "network",
    ]
    assert [step.progress_fraction for step in steps] == [0.4, 0.6, 0.8, 1.0]
    assert steps[0].resolve_fn() == {"energy": []}
    assert steps[3].resolve_fn() == {"strands": []}
    assert calls == ["energy", "network"]


def test_run_pipeline_stage_sequence_stops_after_matching_stage():
    results: dict[str, object] = {"parameters": {}}
    calls: list[str] = []
    run_context = _DummyRunContext()

    early = run_pipeline_stage_sequence(
        results,
        steps=[
            PipelineStageStep(
                result_key="energy_data",
                stage_name="energy",
                progress_fraction=0.4,
                resolve_fn=lambda: calls.append("energy") or {"energy": []},
            ),
            PipelineStageStep(
                result_key="vertices",
                stage_name="vertices",
                progress_fraction=0.6,
                resolve_fn=lambda: calls.append("vertices") or {"positions": []},
            ),
        ],
        progress_callback=None,
        stop_after="energy",
        run_context=run_context,
    )

    assert calls == ["energy"]
    assert early is not None
    assert "vertices" not in early
    assert run_context.stop_after_calls == ["energy"]
