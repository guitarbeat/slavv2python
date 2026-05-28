import typing

if typing.TYPE_CHECKING:
    from .run_ledger import RunContext
    from .stage_handle import StageController

from .models import ProgressEvent, RunSnapshot, StageSnapshot, TaskSnapshot
from .snapshots import emit_progress_event, load_or_create_snapshot, persist_snapshot
from .tracker import (
    PIPELINE_STAGES,
    PREPROCESS_STAGE,
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    STATUS_COMPLETED_TARGET,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    TRACKED_RUN_STAGES,
    atomic_joblib_dump,
    atomic_write_json,
    atomic_write_text,
    build_status_lines,
    fingerprint_array,
    fingerprint_file,
    fingerprint_jsonable,
    load_json_dict,
    load_run_snapshot,
    stable_json_dumps,
)


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
