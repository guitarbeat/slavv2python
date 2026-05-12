"""Utility functions for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from slavv_python.runtime.io import (
    atomic_joblib_dump,
    atomic_write_json,
    atomic_write_text,
    fingerprint_array,
    fingerprint_file,
    fingerprint_jsonable,
)

from .constants import NORMALIZED_DIR


def now_iso() -> str:
    """Current time in ISO 8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def string_or_none(value: Any) -> str | None:
    """Sanitize a string value or return None if empty."""
    if isinstance(value, str) and value.strip():
        return value
    return None


def entity_id_from_path(path: Path) -> str:
    """Derive a unique ID from a file or directory name."""
    return path.name or path.resolve().name


def resolve_python_commit(repo_root: Path) -> str | None:
    """Get the current git commit hash."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            encoding="utf-8",
        )
    except OSError:
        return None
    commit = completed.stdout.strip()
    return commit or None


def normalize_value(value: Any) -> Any:
    """Recursively normalize numpy types and containers for JSON/comparison."""
    if isinstance(value, np.ndarray):
        return normalize_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    return value


def payload_hash(payload: Any) -> str:
    """Generate a stable fingerprint for a potentially complex payload."""
    return fingerprint_jsonable(_hashable_payload_summary(payload))


def _hashable_payload_summary(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return {
            "kind": "ndarray",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": fingerprint_array(array),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _hashable_payload_summary(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_hashable_payload_summary(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_hash_sidecar(path: Path) -> Path:
    """Write a .sha256 sidecar file for a physical file."""
    hash_path = path.with_name(f"{path.name}.sha256")
    atomic_write_text(hash_path, fingerprint_file(path))
    return hash_path


def write_payload_hash_sidecar(path: Path, payload: Any) -> Path:
    """Write a .sha256 sidecar file for a serializable payload."""
    hash_path = path.with_name(f"{path.name}.sha256")
    atomic_write_text(hash_path, payload_hash(payload))
    return hash_path


def write_json_with_hash(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON file and its hash sidecar."""
    atomic_write_json(path, payload)
    write_hash_sidecar(path)
    return path


def write_text_with_hash(path: Path, text: str) -> Path:
    """Write a text file and its hash sidecar."""
    atomic_write_text(path, text)
    write_hash_sidecar(path)
    return path


def write_joblib_with_hash(path: Path, payload: Any) -> Path:
    """Write a joblib dump and its hash sidecar."""
    atomic_joblib_dump(payload, path)
    write_payload_hash_sidecar(path, payload)
    return path


def persist_normalized_payloads(
    dest_run_root: Path,
    *,
    group_name: str,
    payloads: dict[str, Any],
) -> dict[str, str]:
    """Persist normalized checkpoints for comparison."""
    written: dict[str, str] = {}
    normalized_root = dest_run_root / NORMALIZED_DIR / group_name
    normalized_root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        artifact_path = normalized_root / f"{name}.pkl"
        write_joblib_with_hash(artifact_path, payload)
        written[name] = str(artifact_path)
    return written


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records to a JSONL file."""
    import json

    lines = [json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
