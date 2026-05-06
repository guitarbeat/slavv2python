"""Table generation logic for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from source.runtime.run_tracking.io import atomic_write_text, stable_json_dumps

from .io import (
    write_hash_sidecar,
)

if TYPE_CHECKING:
    from pathlib import Path


def _coerce_table_cell(value: Any) -> Any:
    from .execution import _normalize_param_value

    normalized = _normalize_param_value(value)
    if isinstance(normalized, float) and normalized != normalized:
        return None
    if isinstance(normalized, (list, dict)):
        return stable_json_dumps(normalized)
    return normalized


def _persist_table_records(
    tables_root: Path,
    *,
    table_name: str,
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not records:
        return None

    from .execution import _normalize_param_value

    normalized_records = [
        cast("dict[str, Any]", _normalize_param_value(dict(record))) for record in records
    ]
    frame = pd.json_normalize(normalized_records, sep=".")
    frame = frame.reindex(sorted(frame.columns), axis=1)
    frame = frame.apply(lambda column: column.map(_coerce_table_cell))

    jsonl_path = tables_root / f"{table_name}.jsonl"
    csv_path = tables_root / f"{table_name}.csv"

    row_payloads = [
        {str(key): _coerce_table_cell(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]
    jsonl_text = "".join(f"{stable_json_dumps(row)}\n" for row in row_payloads)
    atomic_write_text(jsonl_path, jsonl_text)
    write_hash_sidecar(jsonl_path)

    csv_text = frame.to_csv(index=False)
    atomic_write_text(csv_path, csv_text)
    write_hash_sidecar(csv_path)

    return {
        "name": table_name,
        "row_count": len(row_payloads),
        "column_count": len(frame.columns),
        "columns": [str(column) for column in frame.columns.tolist()],
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
    }


# ... (I'll include all the _build_*_tables functions here)
