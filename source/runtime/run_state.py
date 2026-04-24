"""File-backed run state for resumable SLAVV processing."""

from __future__ import annotations

from ._run_state.constants import (
    PIPELINE_STAGES,
    PREPROCESS_STAGE,
    STAGE_WEIGHTS,
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    STATUS_COMPLETED_TARGET,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
)
from ._run_state.context import RunContext, StageController
from ._run_state.io import (
    atomic_joblib_dump,
    atomic_write_json,
    atomic_write_text,
    fingerprint_array,
    fingerprint_file,
    fingerprint_jsonable,
    load_json_dict,
    load_run_snapshot,
    stable_json_dumps,
)
from ._run_state.models import ProgressEvent, RunSnapshot, StageSnapshot, TaskSnapshot
from ._run_state.status import build_status_lines, target_stage_progress

__all__ = [
    "PIPELINE_STAGES",
    "PREPROCESS_STAGE",
    "STAGE_WEIGHTS",
    "STATUS_BLOCKED",
    "STATUS_COMPLETED",
    "STATUS_COMPLETED_TARGET",
    "STATUS_FAILED",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "ProgressEvent",
    "RunContext",
    "RunSnapshot",
    "StageController",
    "StageSnapshot",
    "TaskSnapshot",
    "atomic_joblib_dump",
    "atomic_write_json",
    "atomic_write_text",
    "build_status_lines",
    "fingerprint_array",
    "fingerprint_file",
    "fingerprint_jsonable",
    "load_json_dict",
    "load_run_snapshot",
    "stable_json_dumps",
    "target_stage_progress",
]
