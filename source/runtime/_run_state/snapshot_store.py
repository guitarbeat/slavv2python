"""Helpers for loading, creating, persisting, and emitting run snapshots."""

from __future__ import annotations

import copy
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable

from .constants import STATUS_PENDING
from .io import (
    _ensure_stage_map,
    _normalize_for_json,
    atomic_write_json,
    load_run_snapshot,
)
from .models import ProgressEvent, RunSnapshot, StageSnapshot, _now_iso

if TYPE_CHECKING:
    from .layout import RunLayout


def load_or_create_snapshot(
    layout: RunLayout,
    *,
    input_fingerprint: str,
    params_fingerprint: str,
    target_stage: str | None,
    provenance: dict[str, Any],
) -> RunSnapshot:
    """Load an existing snapshot or create a fresh one for the resolved layout."""
    layout.ensure_directories()

    existing = load_run_snapshot(layout.snapshot_path)

    if existing is not None:
        existing.stages = _ensure_stage_map(existing.stages)
        if input_fingerprint and not existing.input_fingerprint:
            existing.input_fingerprint = input_fingerprint
        if params_fingerprint and not existing.params_fingerprint:
            existing.params_fingerprint = params_fingerprint
        if target_stage is not None:
            existing.target_stage = target_stage
        if provenance:
            existing.provenance.update(_normalize_for_json(provenance))
        existing.updated_at = _now_iso()
        return existing

    return RunSnapshot(
        run_id=uuid.uuid4().hex[:12],
        input_fingerprint=input_fingerprint,
        params_fingerprint=params_fingerprint,
        status=STATUS_PENDING,
        target_stage=target_stage or "network",
        stages=_ensure_stage_map(),
        provenance=_normalize_for_json({"layout": "structured", **provenance}),
    )


def persist_snapshot(snapshot: RunSnapshot, snapshot_path, *, start_time: float) -> None:
    """Persist a snapshot to disk while keeping elapsed seconds monotonic."""
    snapshot.elapsed_seconds = max(snapshot.elapsed_seconds, time.time() - start_time)
    snapshot.updated_at = _now_iso()
    atomic_write_json(snapshot_path, snapshot.to_dict())


def emit_progress_event(
    snapshot: RunSnapshot,
    event_callback: Callable[[ProgressEvent], None] | None,
    *,
    stage: str,
    status: str,
    detail: str = "",
) -> None:
    """Emit a deep-copied progress event if a callback is configured."""
    if event_callback is None:
        return
    stage_snapshot = snapshot.stages.get(stage, StageSnapshot(name=stage))
    payload = ProgressEvent(
        stage=stage,
        status=status,
        overall_progress=snapshot.overall_progress,
        stage_progress=stage_snapshot.progress,
        detail=detail,
        resumed=stage_snapshot.resumed,
        snapshot=copy.deepcopy(snapshot),
    )
    event_callback(payload)


__all__ = [
    "emit_progress_event",
    "load_or_create_snapshot",
    "persist_snapshot",
]
