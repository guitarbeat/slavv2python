"""Run State package: ledger, stage handles, snapshots, and IO helpers.

Import public names from this package. Prefer owning modules for new code:

* ``run_ledger.RunContext`` / ``stage_handle.StageController`` — lifecycle
* ``io`` — atomic write/load and fingerprints
* ``models`` — snapshot dataclasses
* ``status`` — status line rendering
* ``engine.constants`` — stage names and status string constants

There is no ``tracker`` module; that name was a shallow re-export barrel.
"""

from __future__ import annotations

import typing

from slavv_python.engine.constants import (
    PIPELINE_STAGES,
    PREPROCESS_STAGE,
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    STATUS_COMPLETED_TARGET,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    TRACKED_RUN_STAGES,
)
from slavv_python.engine.state.io import (
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
from slavv_python.engine.state.models import ProgressEvent, RunSnapshot, StageSnapshot, TaskSnapshot
from slavv_python.engine.state.snapshots import (
    emit_progress_event,
    load_or_create_snapshot,
    persist_snapshot,
)
from slavv_python.engine.state.status import build_status_lines

if typing.TYPE_CHECKING:
    from slavv_python.engine.state.run_ledger import RunContext
    from slavv_python.engine.state.stage_handle import StageController


def __getattr__(name: str) -> typing.Any:
    if name == "RunContext":
        from slavv_python.engine.state.run_ledger import RunContext as _RunContext

        return _RunContext
    if name == "StageController":
        from slavv_python.engine.state.stage_handle import StageController as _StageController

        return _StageController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PIPELINE_STAGES",
    "PREPROCESS_STAGE",
    "STATUS_BLOCKED",
    "STATUS_COMPLETED",
    "STATUS_COMPLETED_TARGET",
    "STATUS_FAILED",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "TRACKED_RUN_STAGES",
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
    "emit_progress_event",
    "fingerprint_array",
    "fingerprint_file",
    "fingerprint_jsonable",
    "load_json_dict",
    "load_or_create_snapshot",
    "load_run_snapshot",
    "persist_snapshot",
    "stable_json_dumps",
]
