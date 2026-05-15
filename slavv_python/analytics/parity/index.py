"""Index and layout management for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from slavv_python.runtime.io import atomic_write_text, stable_json_dumps

from .constants import EXPERIMENT_INDEX_PATH, EXPERIMENT_ROOT_SUBDIRS


def resolve_experiment_root(path: Path) -> Path | None:
    """Find the structured experiment root by looking for standard subdirs."""
    resolved = path.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name in EXPERIMENT_ROOT_SUBDIRS:
            return candidate.parent
        if all((candidate / subdir).is_dir() for subdir in EXPERIMENT_ROOT_SUBDIRS):
            return candidate
    return None


def ensure_experiment_root_layout(root: Path) -> None:
    """Ensure the standard directory structure exists."""
    for subdir in EXPERIMENT_ROOT_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load records from an append-only JSONL index."""
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(cast("dict[str, Any]", payload))
    return records


def upsert_index_record(root: Path | None, payload: dict[str, Any]) -> None:
    """Upsert a record into the central experiment index."""
    if root is None:
        return
    ensure_experiment_root_layout(root)
    index_path = root / EXPERIMENT_INDEX_PATH

    # Identify unique record
    payload_id = str(
        payload.get("id")
        or payload.get("run_id")
        or (Path(payload["path"]).name if "path" in payload else None)
        or (Path(payload["run_root"]).name if "run_root" in payload else "unknown")
    )
    payload_kind = str(payload.get("kind", "artifact"))

    retained: list[dict[str, Any]] = []
    for existing in load_jsonl_records(index_path):
        if str(existing.get("id")) == payload_id and str(existing.get("kind")) == payload_kind:
            continue
        retained.append(existing)

    retained.append(payload)
    atomic_write_text(
        index_path,
        "".join(f"{stable_json_dumps(record)}\n" for record in retained),
    )


def deduplicate_index_records(root: Path | None, dry_run: bool = False) -> list[dict[str, Any]]:
    """Deduplicate index records and remove stale paths from index.jsonl.

    Keeps only the latest record for each (id, kind) combination and
    filters out records with non-existent paths.
    """
    if root is None:
        return []
    index_path = root / EXPERIMENT_INDEX_PATH
    if not index_path.is_file():
        return []

    records = load_jsonl_records(index_path)
    retained: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    # Map (id, kind) to the index of the latest record
    latest_indices = {}
    for i, rec in enumerate(records):
        rec_id = str(
            rec.get("id")
            or rec.get("run_id")
            or (Path(rec["path"]).name if "path" in rec else None)
            or (Path(rec["run_root"]).name if "run_root" in rec else "unknown")
        )
        rec_kind = str(rec.get("kind", "artifact"))
        key = (rec_id, rec_kind)
        latest_indices[key] = i

    for i, rec in enumerate(records):
        rec_id = str(
            rec.get("id")
            or rec.get("run_id")
            or (Path(rec["path"]).name if "path" in rec else None)
            or (Path(rec["run_root"]).name if "run_root" in rec else "unknown")
        )
        rec_kind = str(rec.get("kind", "artifact"))
        key = (rec_id, rec_kind)

        if latest_indices[key] != i:
            removed.append(rec)
            continue

        # Check if path exists if present
        path_val = rec.get("path") or rec.get("run_root")
        if path_val:
            p = Path(path_val)
            if not p.exists():
                removed.append(rec)
                continue

        retained.append(rec)

    if not dry_run and removed:
        atomic_write_text(
            index_path,
            "".join(f"{stable_json_dumps(record)}\n" for record in retained),
        )

    return removed
