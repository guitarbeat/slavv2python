"""File-backed run tracking for resumable SLAVV processing."""

from __future__ import annotations

from .run_tracking.constants import (
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
from .run_tracking.context import RunContext, StageController
from .run_tracking.io import (
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
from .run_tracking.models import ProgressEvent, RunSnapshot, StageSnapshot, TaskSnapshot
from .run_tracking.status import build_status_lines, target_stage_progress

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
