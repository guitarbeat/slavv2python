from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .constants import (
    PIPELINE_STAGES,
    PREPROCESS_STAGE,
    STAGE_WEIGHTS,
    STATUS_COMPLETED,
    STATUS_PENDING,
    TRACKED_RUN_STAGES,
)
from .models import RunSnapshot, StageSnapshot, _now_iso


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return [_normalize_for_json(v) for v in sorted(value)]
    return value


def stable_json_dumps(value: Any) -> str:
    """Serialize a value with deterministic ordering."""
    return json.dumps(_normalize_for_json(value), sort_keys=True, separators=(",", ":"))


def fingerprint_jsonable(value: Any) -> str:
    """Create a content hash for JSON-like data."""
    payload = stable_json_dumps(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fingerprint_array(array: np.ndarray) -> str:
    """Create a content hash for a numpy array."""
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def fingerprint_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Create a content hash for a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            if chunk := handle.read(chunk_size):
                hasher.update(chunk)
            else:
                break
    return hasher.hexdigest()


def _replace_with_retry(
    tmp_name: str, target: Path, *, attempts: int = 20, delay: float = 0.25
) -> None:
    """Retry atomic replacement to tolerate transient Windows file locks."""
    last_error = None
    for attempt in range(attempts):
        try:
            os.replace(tmp_name, target)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
    if last_error is not None:
        raise last_error


def atomic_write_json(path: str | Path, data: Any) -> None:
    """Atomically write JSON content."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_normalize_for_json(data), handle, indent=2, sort_keys=True)
        _replace_with_retry(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_joblib_dump(value: Any, path: str | Path) -> None:
    """Atomically write a joblib artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name, suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(value, tmp_name)
        _replace_with_retry(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_run_snapshot(path_or_dir: str | Path) -> RunSnapshot | None:
    """Load a run snapshot from a file or directory if present."""
    path = Path(path_or_dir)
    candidates = []
    if path.is_dir():
        candidates.extend(
            [
                path / "run_snapshot.json",
                path / "99_Metadata" / "run_snapshot.json",
                path / "metadata" / "run_snapshot.json",
            ]
        )
    else:
        candidates.append(path)

    for candidate in candidates:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as handle:
                return RunSnapshot.from_dict(json.load(handle))
    return None


def _ensure_stage_map(existing: dict[str, StageSnapshot] | None = None) -> dict[str, StageSnapshot]:
    stages = {name: StageSnapshot(name=name) for name in TRACKED_RUN_STAGES}
    if existing:
        stages |= existing
        for name in TRACKED_RUN_STAGES:
            stages.setdefault(name, StageSnapshot(name=name))
    return stages


def load_legacy_run_snapshot(
    checkpoint_dir: str | Path, *, target_stage: str = "network"
) -> RunSnapshot | None:
    """Inspect legacy checkpoint directories without mutating them."""
    checkpoints_dir = Path(checkpoint_dir)
    checkpoint_paths = {
        stage: checkpoints_dir / f"checkpoint_{stage}.pkl" for stage in PIPELINE_STAGES
    }
    if not any(path.exists() for path in checkpoint_paths.values()):
        return None

    snapshot = RunSnapshot(
        run_id=hashlib.sha1(str(checkpoints_dir.resolve()).encode("utf-8")).hexdigest()[:12],
        target_stage=target_stage,
        status=STATUS_PENDING,
        stages=_ensure_stage_map(),
        provenance={"layout": "legacy"},
    )
    preprocess_stage = snapshot.stages[PREPROCESS_STAGE]
    preprocess_stage.status = STATUS_COMPLETED
    preprocess_stage.progress = 1.0
    preprocess_stage.units_total = 1
    preprocess_stage.units_completed = 1
    preprocess_stage.resumed = True
    for stage, path in checkpoint_paths.items():
        if not path.exists():
            continue
        stage_snapshot = snapshot.stages[stage]
        stage_snapshot.status = STATUS_COMPLETED
        stage_snapshot.progress = 1.0
        stage_snapshot.units_total = 1
        stage_snapshot.units_completed = 1
        stage_snapshot.resumed = True
        stage_snapshot.artifacts["checkpoint"] = str(path)
        stage_snapshot.completed_at = _now_iso()
    total = STAGE_WEIGHTS[PREPROCESS_STAGE] + sum(STAGE_WEIGHTS[stage] for stage in PIPELINE_STAGES)
    snapshot.overall_progress = (
        sum(STAGE_WEIGHTS[stage] * snapshot.stages[stage].progress for stage in TRACKED_RUN_STAGES)
        / total
    )
    return snapshot
