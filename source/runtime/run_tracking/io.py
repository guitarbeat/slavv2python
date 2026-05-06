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

from .constants import TRACKED_RUN_STAGES
from .models import RunSnapshot, StageSnapshot


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
            chunk = handle.read(chunk_size)
            if chunk:
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


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically write UTF-8 text content."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        _replace_with_retry(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_json_dict(path: str | Path) -> dict[str, Any] | None:
    """Load a JSON file when it exists and contains an object payload."""
    target = Path(path)
    if not target.exists():
        return None
    try:
        with open(target, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


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
        candidates.append(path / "99_Metadata" / "run_snapshot.json")
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
        stages.update(existing)
        for name in TRACKED_RUN_STAGES:
            stages.setdefault(name, StageSnapshot(name=name))
    return stages
