"""Freshness checks for Energy exact-proof inputs and historical reports."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from slavv_python.engine.state import load_json_dict

from .constants import (
    ANALYSIS_DIR,
    CHECKPOINTS_DIR,
    PARITY_JOB_MANIFEST_PATH,
    RUN_SNAPSHOT_PATH,
    WRITER_LEASE_PATH,
)

_ENERGY_STAGE_NAME = "energy"

if TYPE_CHECKING:
    from pathlib import Path

_ENERGY_CHECKPOINT = "checkpoint_energy.pkl"
_REPORT_PATTERNS = ("exact_proof*.json", "exact_mismatch*.json")


def build_energy_proof_evidence(run_root: Path) -> dict[str, Any]:
    """Describe whether a run has current, completed Energy proof inputs.

    Historical proof reports are never removed.  They are marked stale when they
    cannot have been produced from the current Energy checkpoint or when the
    current Energy attempt itself is not usable.
    """
    root = run_root.expanduser().resolve()
    checkpoint = root / CHECKPOINTS_DIR / _ENERGY_CHECKPOINT
    snapshot_path = root / RUN_SNAPSHOT_PATH
    snapshot = load_json_dict(snapshot_path) if snapshot_path.is_file() else None
    failures: list[str] = []

    if not checkpoint.is_file():
        failures.append("missing_energy_checkpoint")
    if not isinstance(snapshot, dict):
        failures.append("missing_run_snapshot")
        energy_stage: dict[str, Any] = {}
    else:
        energy_stage = _resolve_energy_stage(snapshot, root)
        status = energy_stage.get("status")
        if status != "completed":
            failures.append(
                f"energy_stage_status_{status if isinstance(status, str) else 'missing'}"
            )

    checkpoint_timestamp = _mtime_utc(checkpoint) if checkpoint.is_file() else None
    started_at = _parse_timestamp(energy_stage.get("started_at"))
    if checkpoint_timestamp is not None and started_at is not None:
        if checkpoint_timestamp < started_at:
            failures.append("energy_checkpoint_predates_latest_start")
    elif checkpoint_timestamp is not None:
        failures.append("missing_energy_stage_started_at")

    valid = not failures
    return {
        "schema_version": 1,
        "run_root": str(root),
        "valid": valid,
        "failures": failures,
        "energy_checkpoint": {
            "path": str(checkpoint),
            "exists": checkpoint.is_file(),
            "modified_at": _format_timestamp(checkpoint_timestamp),
        },
        "run_snapshot": {
            "path": str(snapshot_path),
            "exists": snapshot_path.is_file(),
            "status": snapshot.get("status") if isinstance(snapshot, dict) else None,
            "energy_status": energy_stage.get("status"),
            "energy_started_at": energy_stage.get("started_at"),
            "energy_updated_at": energy_stage.get("updated_at"),
        },
        "writer_lease": _operational_record(root / WRITER_LEASE_PATH),
        "parity_job": _operational_record(root / PARITY_JOB_MANIFEST_PATH),
        "historical_reports": _historical_reports(
            root,
            checkpoint_timestamp=checkpoint_timestamp,
            evidence_valid=valid,
        ),
    }


def require_energy_proof_evidence(run_root: Path) -> dict[str, Any]:
    """Return current Energy evidence or reject stale proof inputs clearly."""
    report = build_energy_proof_evidence(run_root)
    if not report["valid"]:
        reasons = ", ".join(report["failures"])
        raise ValueError(
            "Energy proof evidence is stale; rerun or restore a completed Energy checkpoint "
            f"before proving parity ({reasons})."
        )
    return report


def _resolve_energy_stage(snapshot: dict[str, Any], run_root: Path) -> dict[str, Any]:
    """Return the Energy stage record from snapshot stages or parity stage_metrics."""
    stages = snapshot.get("stages")
    energy_stage = stages.get(_ENERGY_STAGE_NAME, {}) if isinstance(stages, dict) else {}
    if not isinstance(energy_stage, dict):
        energy_stage = {}

    if energy_stage.get("status") == "completed":
        return energy_stage

    stage_metrics = snapshot.get("stage_metrics")
    metrics = stage_metrics.get(_ENERGY_STAGE_NAME, {}) if isinstance(stage_metrics, dict) else {}
    if not isinstance(metrics, dict) or metrics.get("status") != "completed":
        return energy_stage

    lease = load_json_dict(run_root / WRITER_LEASE_PATH)
    started_at = energy_stage.get("started_at")
    if (
        (not isinstance(started_at, str) or not started_at)
        and isinstance(lease, dict)
        and isinstance(lease.get("started_at"), str)
    ):
        started_at = lease["started_at"]

    updated_at = energy_stage.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at:
        completed_at = metrics.get("completed_at")
        if isinstance(completed_at, str) and completed_at:
            updated_at = completed_at
        elif isinstance(lease, dict) and isinstance(lease.get("updated_at"), str):
            updated_at = lease["updated_at"]

    return {
        "status": metrics.get("status"),
        "started_at": started_at,
        "updated_at": updated_at,
        "completed_at": metrics.get("completed_at"),
    }


def _operational_record(path: Path) -> dict[str, Any]:
    payload = load_json_dict(path) if path.is_file() else None
    return {
        "path": str(path),
        "exists": path.is_file(),
        "status": payload.get("status") if isinstance(payload, dict) else None,
        "stage": payload.get("stage") if isinstance(payload, dict) else None,
    }


def _historical_reports(
    run_root: Path,
    *,
    checkpoint_timestamp: datetime | None,
    evidence_valid: bool,
) -> list[dict[str, str | bool]]:
    reports: dict[Path, None] = {}
    analysis_dir = run_root / ANALYSIS_DIR
    for pattern in _REPORT_PATTERNS:
        for path in analysis_dir.glob(pattern):
            reports[path] = None

    result: list[dict[str, str | bool]] = []
    for path in sorted(reports):
        report_mtime = _mtime_utc(path)
        stale = not evidence_valid
        if stale:
            reason = "current_energy_evidence_invalid"
        elif checkpoint_timestamp is not None and report_mtime < checkpoint_timestamp:
            stale = True
            reason = "predates_current_energy_checkpoint"
        else:
            reason = ""
        result.append(
            {
                "path": str(path),
                "stale": stale,
                "stale_reason": reason,
            }
        )
    return result


def _mtime_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_timestamp(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


__all__ = ["build_energy_proof_evidence", "require_energy_proof_evidence"]
