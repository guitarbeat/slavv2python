from __future__ import annotations

import numpy as np
from dev.tests.support.payload_builders import build_processing_results

from source.workflows.pipeline_results import (
    finalize_pipeline_results,
    stop_after_stage_if_requested,
)


class _DummyRunContext:
    def __init__(self) -> None:
        self.stop_after_calls: list[str | None] = []

    def finalize_run(self, *, stop_after: str | None = None) -> None:
        self.stop_after_calls.append(stop_after)


def test_finalize_pipeline_results_returns_plain_dict_payload():
    payload = build_processing_results(overrides={"metadata": {"source": "workflow"}})

    finalized = finalize_pipeline_results(payload)

    assert finalized["metadata"] == {"source": "workflow"}
    assert np.array_equal(finalized["vertices"]["positions"], payload["vertices"]["positions"])


def test_stop_after_stage_if_requested_finalizes_run_context():
    payload = build_processing_results()
    run_context = _DummyRunContext()

    early = stop_after_stage_if_requested("energy", "energy", payload, run_context)

    assert early is not None
    assert run_context.stop_after_calls == ["energy"]
    assert "vertices" in early


def test_stop_after_stage_if_requested_skips_other_stages():
    payload = build_processing_results()
    run_context = _DummyRunContext()

    early = stop_after_stage_if_requested("network", "energy", payload, run_context)

    assert early is None
    assert run_context.stop_after_calls == []
