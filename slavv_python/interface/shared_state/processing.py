"""Helpers for processing-page state and result bookkeeping."""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import TYPE_CHECKING, Any

from slavv_python.engine.state.io import fingerprint_jsonable
from slavv_python.schema.app_run import AppRunState
from slavv_python.schema.results import PipelineResult, normalize_pipeline_result

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

    from slavv_python.engine.state import RunSnapshot


def build_processing_run_dir(upload_bytes: bytes, validated_params: dict[str, object]) -> str:
    """Return a stable run directory per input file and validated parameter set."""
    file_hash = hashlib.md5(upload_bytes).hexdigest()[:12]
    params_hash = fingerprint_jsonable(validated_params)[:12]
    return os.path.join(tempfile.gettempdir(), "slavv_runs", f"{file_hash}_{params_hash}")


def load_processing_snapshot(
    session_state: Mapping[str, Any],
    *,
    snapshot_loader,
) -> RunSnapshot | None:
    """Load the current run snapshot from session state when a run dir exists."""
    run_dir = session_state.get("current_run_dir")
    if not run_dir:
        return None
    return snapshot_loader(run_dir)


def summarize_processing_metrics(
    processing_results: AppRunState | PipelineResult | Mapping[str, Any],
) -> dict[str, int]:
    """Return lightweight post-run counts for processing-page metrics."""
    pipeline = AppRunState.from_value(processing_results).pipeline
    return {
        "vertices": len(pipeline.vertices.positions) if pipeline.vertices is not None else 0,
        "edges": len(pipeline.edges.traces) if pipeline.edges is not None else 0,
        "strands": len(pipeline.network.strands) if pipeline.network is not None else 0,
        "bifurcations": len(pipeline.network.bifurcations) if pipeline.network is not None else 0,
    }


def store_processing_session_state(
    session_state: MutableMapping[str, Any],
    *,
    results: PipelineResult | Mapping[str, Any],
    validated_params: dict[str, Any],
    image_shape: tuple[int, ...],
    dataset_name: str,
    run_dir: str | None,
    final_snapshot: RunSnapshot | None,
) -> None:
    """Persist completed processing state and clear derived stale session keys."""
    pipeline = (
        results if isinstance(results, PipelineResult) else normalize_pipeline_result(dict(results))
    )
    session_state["processing_results"] = AppRunState(
        pipeline=pipeline,
        image_shape=image_shape,
        dataset_name=dataset_name,
        run_dir=run_dir,
    )
    session_state["parameters"] = validated_params
    session_state["image_shape"] = image_shape
    session_state["dataset_name"] = dataset_name
    session_state["current_run_dir"] = run_dir
    session_state["run_snapshot"] = final_snapshot.to_dict() if final_snapshot is not None else None
    session_state.pop("curation_baseline_counts", None)
    session_state.pop("last_curation_mode", None)
    session_state.pop("share_report_prepared_signature", None)


__all__ = [
    "build_processing_run_dir",
    "load_processing_snapshot",
    "store_processing_session_state",
    "summarize_processing_metrics",
]
