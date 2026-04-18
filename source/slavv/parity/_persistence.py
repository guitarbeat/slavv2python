"""Shared parity-local persistence helpers for report-style artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from slavv.runtime.run_state import atomic_write_text, load_json_dict


def _json_default(value: Any) -> Any:
    """Serialize numpy-heavy parity payloads to plain JSON primitives."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_default_with_string_fallback(value: Any) -> Any:
    """Serialize report payloads, falling back to strings for unknown values."""
    try:
        return _json_default(value)
    except TypeError:
        return str(value)


def load_json_dict_or_empty(path: str | Path) -> dict[str, Any]:
    """Load a JSON object file, returning an empty mapping on failure."""
    payload = load_json_dict(path)
    return {} if payload is None else payload


def write_json_file(
    path: str | Path,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
    default=None,
) -> Path:
    """Serialize and atomically persist a JSON artifact."""
    target = Path(path)
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys, default=default)
    atomic_write_text(target, text + "\n")
    return target


def write_lines_file(path: str | Path, lines: list[str]) -> Path:
    """Persist a newline-delimited UTF-8 text file."""
    target = Path(path)
    atomic_write_text(target, "\n".join(lines))
    return target


def write_text_file(path: str | Path, text: str) -> Path:
    """Persist a UTF-8 text artifact verbatim."""
    target = Path(path)
    atomic_write_text(target, text)
    return target


def write_kv_tsv(path: str | Path, rows: dict[str, Any]) -> Path:
    """Persist a deterministic key/value TSV file."""
    target = Path(path)
    tsv_lines = ["metric\tvalue"]
    tsv_lines.extend(f"{key}\t{value}" for key, value in sorted(rows.items()))
    atomic_write_text(target, "\n".join(tsv_lines) + "\n")
    return target


def infer_date_str(run_name: str) -> str:
    """Infer a display date from the leading run-name token when possible."""
    date_str = "Unknown"
    date_part = run_name.split("_")[0]
    if len(date_part) == 8 and date_part.isdigit():
        try:
            date_obj = datetime.strptime(date_part, "%Y%m%d")
            date_str = date_obj.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return date_str
