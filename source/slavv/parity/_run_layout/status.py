from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from slavv.runtime import load_run_snapshot
from slavv.runtime.run_state import atomic_write_json, load_json_dict

from .paths import (
    is_aggregate_run_container,
    list_aggregate_child_runs,
    resolve_run_layout,
)

_STATUS_STATES = {"completed", "failed", "incomplete", "superseded", "archived"}
_STATUS_RETENTION = {"keep", "eligible_for_cleanup", "archive"}
_STATUS_QUALITY = {"pass", "fail", "partial", "unknown"}


def _read_json_file(path: Path) -> dict[str, Any] | None:
    return load_json_dict(path)


def _normalize_status_payload(status: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "state": str(status.get("state", "incomplete") or "incomplete"),
        "retention": str(status.get("retention", "eligible_for_cleanup") or "eligible_for_cleanup"),
        "quality_gate": str(status.get("quality_gate", "unknown") or "unknown"),
    }
    for key in ("supersedes", "superseded_by", "notes"):
        value = status.get(key)
        if value not in (None, ""):
            payload[key] = str(value)
    return payload


def validate_run_status_payload(status: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a run status payload."""
    payload = _normalize_status_payload(status)
    if payload["state"] not in _STATUS_STATES:
        raise ValueError(f"Unsupported run status state: {payload['state']}")
    if payload["retention"] not in _STATUS_RETENTION:
        raise ValueError(f"Unsupported run status retention: {payload['retention']}")
    if payload["quality_gate"] not in _STATUS_QUALITY:
        raise ValueError(f"Unsupported run status quality gate: {payload['quality_gate']}")
    return payload


def read_run_status(run_dir: Path) -> dict[str, Any] | None:
    """Read managed lifecycle metadata from 99_Metadata/status.json when present."""
    metadata_dir = resolve_run_layout(run_dir)["metadata_dir"]
    payload = _read_json_file(metadata_dir / "status.json")
    return None if payload is None else validate_run_status_payload(payload)


def write_run_status(run_dir: Path, status: dict[str, Any]) -> Path:
    """Persist deterministic lifecycle metadata for a run."""
    metadata_dir = resolve_run_layout(run_dir)["metadata_dir"]
    status_file = metadata_dir / "status.json"
    atomic_write_json(status_file, validate_run_status_payload(status))
    return status_file


def infer_quality_gate(run_dir: Path) -> str:
    """Infer quality gate from explicit status or comparison report heuristics."""
    explicit = read_run_status(run_dir)
    if explicit is not None:
        return str(explicit["quality_gate"])

    report = _read_json_file(resolve_run_layout(run_dir)["report_file"])
    if report is None:
        return "unknown"

    parity_gate = report.get("parity_gate")
    if isinstance(parity_gate, dict) and "passed" in parity_gate:
        if bool(parity_gate.get("passed")):
            return "pass"
        component_flags = [
            parity_gate.get("vertices_exact"),
            parity_gate.get("edges_exact"),
            parity_gate.get("strands_exact"),
        ]
        if any(flag is True for flag in component_flags):
            return "partial"
        if any(flag is False for flag in component_flags):
            return "fail"

    comparison_flags: list[bool] = []
    for section in ("vertices", "edges", "strands", "network"):
        payload = report.get(section)
        if not isinstance(payload, dict):
            continue
        if "matches_exactly" in payload:
            comparison_flags.append(bool(payload["matches_exactly"]))
            continue
        if "exact_match" in payload:
            comparison_flags.append(bool(payload["exact_match"]))

    if comparison_flags:
        if all(comparison_flags):
            return "pass"
        if any(comparison_flags):
            return "partial"
        return "fail"
    return "unknown"


def _infer_state_from_snapshot(run_dir: Path) -> str | None:
    snapshot = load_run_snapshot(run_dir)
    if snapshot is None:
        return None
    status = str(snapshot.status or "").lower()
    if status in {"completed", "completed_target"}:
        return "completed"
    if status in {"failed", "resume_blocked"}:
        return "failed"
    return "incomplete" if status in {"pending", "running"} else None


def _infer_state_from_artifacts(run_dir: Path) -> str | None:
    layout = resolve_run_layout(run_dir)
    if layout["report_file"].exists() or layout["summary_file"].exists():
        return "completed"
    if layout["python_dir"].exists() or layout["matlab_dir"].exists():
        return "incomplete"
    return None


def aggregate_container_rollup(container_dir: Path) -> dict[str, Any]:
    """Summarize aggregate run_* children without treating the container as a managed run."""
    child_runs = list_aggregate_child_runs(container_dir)
    child_statuses = [infer_run_status(child) for child in child_runs]
    states = [status["state"] for status in child_statuses]
    if child_statuses and all(state == "completed" for state in states):
        state = "completed"
    elif "failed" in states:
        state = "failed"
    else:
        state = "incomplete"

    quality_gate = "unknown"
    if child_statuses:
        qualities = {status["quality_gate"] for status in child_statuses}
        if len(qualities) == 1:
            quality_gate = next(iter(qualities))
        elif "pass" in qualities or "partial" in qualities:
            quality_gate = "partial"

    return {
        "state": state,
        "retention": "eligible_for_cleanup",
        "quality_gate": quality_gate,
        "child_runs": [str(child) for child in child_runs],
        "child_states": states,
    }


def infer_run_status(run_dir: Path, *, pointer_targeted: bool = False) -> dict[str, Any]:
    """Infer lifecycle metadata using explicit status, snapshots, artifacts, then aggregate rollup."""
    explicit = read_run_status(run_dir)
    if explicit is not None:
        inferred = dict(explicit)
    elif is_aggregate_run_container(run_dir):
        inferred = aggregate_container_rollup(run_dir)
    else:
        inferred = {
            "state": _infer_state_from_snapshot(run_dir)
            or _infer_state_from_artifacts(run_dir)
            or "incomplete",
            "retention": "eligible_for_cleanup",
            "quality_gate": infer_quality_gate(run_dir),
        }
    if pointer_targeted:
        inferred["retention"] = "keep"
    return validate_run_status_payload(inferred)
