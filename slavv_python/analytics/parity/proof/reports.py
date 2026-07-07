"""Report and table generation logic for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

from slavv_python.analytics.parity.constants import (
    ANALYSIS_TABLES_DIR,
    RECORDING_TABLES_INDEX_PATH,
    RUN_MANIFEST_PATH,
    RUN_SNAPSHOT_PATH,
    SUMMARY_JSON_PATH,
    SUMMARY_TEXT_PATH,
)
from slavv_python.analytics.parity.oracle.models import RunCounts  # noqa: TC001
from slavv_python.analytics.parity.proof import counts as _counts
from slavv_python.analytics.parity.utils import (
    entity_id_from_path,
    normalize_value,
    now_iso,
    string_or_none,
    write_hash_sidecar,
    write_json_with_hash,
    write_text_with_hash,
)
from slavv_python.engine.state import atomic_write_text, load_json_dict, stable_json_dumps

if TYPE_CHECKING:
    from pathlib import Path

extract_matlab_counts = _counts.extract_matlab_counts
extract_source_python_counts = _counts.extract_source_python_counts
read_python_counts_from_run = _counts.read_python_counts_from_run


def _coerce_table_cell(value: Any) -> Any:
    normalized = normalize_value(value)
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

    normalized_records = [
        cast("dict[str, Any]", normalize_value(dict(record))) for record in records
    ]
    try:
        from pandas import json_normalize

        frame = json_normalize(normalized_records, sep=".")
    except ImportError:
        from pandas.io.json import json_normalize

        frame = json_normalize(normalized_records, sep=".")
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


def _build_run_snapshot_tables(
    run_root: Path,
    snapshot_payload: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    run_id = string_or_none(snapshot_payload.get("run_id")) or entity_id_from_path(run_root)
    root_row = {
        **{
            str(key): value
            for key, value in snapshot_payload.items()
            if key not in {"stages", "optional_tasks", "artifacts", "errors"}
        },
        "run_root": str(run_root),
        "stage_count": len(snapshot_payload.get("stages", {})),
    }

    stage_rows: list[dict[str, Any]] = []
    stage_artifact_rows: list[dict[str, Any]] = []
    for stage_key, stage_payload in sorted(
        cast("dict[str, dict[str, Any]]", snapshot_payload.get("stages", {})).items()
    ):
        row = {
            "run_root": str(run_root),
            "run_id": run_id,
            "stage_key": str(stage_key),
            **{str(key): value for key, value in dict(stage_payload).items() if key != "artifacts"},
            "artifact_count": len(stage_payload.get("artifacts", {})),
        }
        stage_rows.append(row)
        for artifact_key, artifact_value in sorted(
            cast("dict[str, Any]", stage_payload.get("artifacts", {})).items()
        ):
            stage_artifact_rows.append(
                {
                    "run_root": str(run_root),
                    "run_id": run_id,
                    "stage_key": str(stage_key),
                    "artifact_key": str(artifact_key),
                    "artifact_value": artifact_value,
                }
            )

    return {
        "run_snapshot": [root_row],
        "run_snapshot_stages": stage_rows,
        "run_snapshot_stage_artifacts": stage_artifact_rows,
    }


def persist_recording_tables(run_root: Path) -> dict[str, Any]:
    """Flatten workflow recordings into CSV/JSONL tables."""
    run_root = run_root.expanduser().resolve()
    tables_root = run_root / ANALYSIS_TABLES_DIR
    tables_root.mkdir(parents=True, exist_ok=True)

    table_entries: list[dict[str, Any]] = []
    source_artifacts: list[str] = []

    snapshot_payload = load_json_dict(run_root / RUN_MANIFEST_PATH)
    if snapshot_payload is None:
        snapshot_payload = load_json_dict(run_root / RUN_SNAPSHOT_PATH)
    if snapshot_payload is not None:
        source_path = (
            run_root / RUN_MANIFEST_PATH
            if (run_root / RUN_MANIFEST_PATH).is_file()
            else run_root / RUN_SNAPSHOT_PATH
        )
        source_artifacts.append(str(source_path))
        for table_name, records in _build_run_snapshot_tables(run_root, snapshot_payload).items():
            entry = _persist_table_records(tables_root, table_name=table_name, records=records)
            if entry:
                table_entries.append(entry)

    from slavv_python.analytics.parity.constants import (
        CANDIDATE_PROGRESS_JSONL_PATH,
        EDGE_CANDIDATE_AUDIT_PATH,
    )

    audit_payload = load_json_dict(run_root / EDGE_CANDIDATE_AUDIT_PATH)
    if audit_payload:
        source_artifacts.append(str(run_root / EDGE_CANDIDATE_AUDIT_PATH))
        # root row
        root_row = {str(k): v for k, v in audit_payload.items() if k != "per_origin_summary"}
        entry = _persist_table_records(
            tables_root, table_name="candidate_audit", records=[root_row]
        )
        if entry:
            table_entries.append(entry)

        # per origin summary
        per_origin = audit_payload.get("per_origin_summary", [])
        entry = _persist_table_records(
            tables_root, table_name="candidate_audit_per_origin", records=per_origin
        )
        if entry:
            table_entries.append(entry)

        # origin metrics
        frontier_counts = audit_payload.get("frontier_per_origin_candidate_counts", {})
        metric_records = []
        for origin_idx, count in frontier_counts.items():
            metric_records.append(
                {
                    "origin_index": int(origin_idx),
                    "metric_name": "frontier_per_origin_candidate_counts",
                    "metric_value": count,
                }
            )
        entry = _persist_table_records(
            tables_root, table_name="candidate_audit_origin_metrics", records=metric_records
        )
        if entry:
            table_entries.append(entry)
    from slavv_python.analytics.parity.constants import CANDIDATE_COVERAGE_JSON_PATH

    coverage_report = load_json_dict(run_root / CANDIDATE_COVERAGE_JSON_PATH)
    if coverage_report:
        source_artifacts.append(str(run_root / CANDIDATE_COVERAGE_JSON_PATH))
        entry = _persist_table_records(
            tables_root, table_name="candidate_coverage_summary", records=[coverage_report]
        )
        if entry:
            table_entries.append(entry)

    if (run_root / CANDIDATE_PROGRESS_JSONL_PATH).is_file():
        source_artifacts.append(str(run_root / CANDIDATE_PROGRESS_JSONL_PATH))
        lines = (run_root / CANDIDATE_PROGRESS_JSONL_PATH).read_text(encoding="utf-8").splitlines()
        import json

        records = [json.loads(line) for line in lines if line.strip()]
        entry = _persist_table_records(
            tables_root, table_name="candidate_progress", records=records
        )
        if entry:
            table_entries.append(entry)

    index_payload = {
        "run_root": str(run_root),
        "created_at": now_iso(),
        "tables_root": str(tables_root),
        "table_count": len(table_entries),
        "source_artifacts": source_artifacts,
        "tables": table_entries,
    }
    write_json_with_hash(run_root / RECORDING_TABLES_INDEX_PATH, index_payload)
    return index_payload


def render_experiment_summary(summary_payload: dict[str, Any]) -> str:
    """Render a human-readable experiment summary."""

    def _format_delta(delta: dict[str, int]) -> str:
        return f"vertices={delta.get('vertices', 0)} edges={delta.get('edges', 0)} strands={delta.get('strands', 0)}"

    matlab = summary_payload.get("matlab_counts", {})
    source_python = summary_payload.get("source_python_counts", {})
    new_python = summary_payload.get("new_python_counts", {})

    return "\n".join(
        [
            "Parity experiment summary",
            f"Source run root: {summary_payload.get('source_run_root')}",
            f"Destination run root: {summary_payload.get('dest_run_root')}",
            f"Rerun from: {summary_payload.get('rerun_from')}",
            "",
            "Counts",
            f"MATLAB: vertices={matlab.get('vertices')} edges={matlab.get('edges')} strands={matlab.get('strands')}",
            f"Source Python: vertices={source_python.get('vertices')} edges={source_python.get('edges')} strands={source_python.get('strands')}",
            f"New Python: vertices={new_python.get('vertices')} edges={new_python.get('edges')} strands={new_python.get('strands')}",
            "",
            "Delta vs MATLAB",
            _format_delta(summary_payload.get("diff_vs_matlab", {})),
            "Delta vs slavv_python Python",
            _format_delta(summary_payload.get("diff_vs_source_python", {})),
        ]
    )


def render_exact_preflight_report(report_payload: dict[str, Any]) -> str:
    """Render a compact exact-preflight report."""
    status = "PASS" if report_payload.get("passed") else "FAIL"
    lines = [
        "Exact preflight report",
        f"Status: {status}",
    ]
    if report_payload.get("forced"):
        lines.append("Force: memory gate override enabled")
    memory_gate = report_payload.get("memory_gate")
    if isinstance(memory_gate, dict):
        required_gb = memory_gate.get("required_bytes", 0) / (1024**3)
        budget_gb = memory_gate.get("budget_bytes", 0) / (1024**3)
        lines.append(
            f"Memory gate: required {required_gb:.2f} GiB <= budget {budget_gb:.2f} GiB "
            f"({'pass' if memory_gate.get('passed') else 'fail'})"
        )
    for error in report_payload.get("errors", []):
        lines.append(f"Error: {error}")
    if "error" in report_payload:
        lines.append(f"Error: {report_payload['error']}")
    for warning in report_payload.get("warnings", []):
        lines.append(f"Warning: {warning}")
    snapshot_status = (report_payload.get("checks") or {}).get("run_snapshot_status")
    if snapshot_status:
        lines.append(f"Run snapshot status: {snapshot_status}")
    return "\n".join(lines)


def persist_experiment_summary(dest_run_root: Path, summary_payload: dict[str, Any]) -> None:
    """Persist the JSON and text summaries."""
    summary_text = render_experiment_summary(summary_payload)
    write_json_with_hash(dest_run_root / SUMMARY_JSON_PATH, summary_payload)
    write_text_with_hash(dest_run_root / SUMMARY_TEXT_PATH, summary_text)


def build_experiment_summary(
    *,
    source_run_root: Path,
    dest_run_root: Path,
    input_file: Path,
    rerun_from: str,
    matlab_counts: RunCounts,
    source_python_counts: RunCounts,
    new_python_counts: RunCounts,
) -> dict[str, Any]:
    """Build the experiment summary payload."""

    def _delta(a: RunCounts, b: RunCounts) -> dict[str, int]:
        return {
            "vertices": a.vertices - b.vertices,
            "edges": a.edges - b.edges,
            "strands": a.strands - b.strands,
        }

    return {
        "source_run_root": str(source_run_root),
        "dest_run_root": str(dest_run_root),
        "input_file": str(input_file),
        "rerun_from": rerun_from,
        "matlab_counts": asdict(matlab_counts),
        "source_python_counts": asdict(source_python_counts),
        "new_python_counts": asdict(new_python_counts),
        "diff_vs_matlab": _delta(new_python_counts, matlab_counts),
        "diff_vs_source_python": _delta(new_python_counts, source_python_counts),
        "timestamp": now_iso(),
    }
