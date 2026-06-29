"""Adaptive Energy mismatch probes and the parity hypothesis circuit breaker."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from slavv_python.analytics.parity.utils import (
    now_iso,
    payload_hash,
    write_json_with_hash,
    write_text_with_hash,
)

SCHEMA_VERSION = 1
REQUEST_PATH = Path("03_Analysis") / "energy_probe_requests.json"
LEDGER_PATH = Path("03_Analysis") / "parity_hypothesis_ledger.json"


@dataclass(frozen=True)
class MismatchGroup:
    """Aggregate count and deterministic representatives for one mismatch class."""

    matlab_scale: int
    python_scale: int
    boundary_class: str
    mismatch_count: int
    first_coordinate_zyx: tuple[int, int, int]
    max_delta_coordinate_zyx: tuple[int, int, int]
    max_abs_delta: float


@dataclass(frozen=True)
class ProbeRequest:
    """One normalized Energy probe request shared by Python and MATLAB."""

    request_id: str
    coordinate_zyx: tuple[int, int, int]
    reason: str
    matlab_scale: int
    python_scale: int
    boundary_class: str


@dataclass(frozen=True)
class ProbeResult:
    """Normalized record emitted by either MATLAB or Python probe execution."""

    request_id: str
    coordinate_zyx: tuple[int, int, int]
    octave: int | None
    chunk_index: int | None
    winner_scale: int
    winner_energy: float
    payload: dict[str, Any]


def build_energy_probe_payload(
    matlab_energy: np.ndarray,
    python_energy: np.ndarray,
    matlab_scales: np.ndarray,
    python_scales: np.ndarray,
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Summarize every mismatch without materializing every mismatch coordinate."""
    _require_same_shape(matlab_energy, python_energy, matlab_scales, python_scales)
    energy_equal = np.equal(matlab_energy, python_energy)
    scale_equal = np.equal(matlab_scales, python_scales)
    mismatch = ~(energy_equal & scale_equal)
    groups: dict[tuple[int, int, str], dict[str, Any]] = defaultdict(dict)
    for linear_index, is_mismatch in enumerate(mismatch.ravel(order="F")):
        if not is_mismatch:
            continue
        coordinate = cast(
            "tuple[int, int, int]",
            tuple(
                int(value) for value in np.unravel_index(linear_index, mismatch.shape, order="F")
            ),
        )
        key = (
            int(matlab_scales[coordinate]),
            int(python_scales[coordinate]),
            _boundary_class(coordinate, mismatch.shape),
        )
        delta = abs(float(matlab_energy[coordinate]) - float(python_energy[coordinate]))
        if not np.isfinite(delta):
            delta = float("inf")
        group = groups[key]
        group["count"] = int(group.get("count", 0)) + 1
        group.setdefault("first_linear", linear_index)
        if delta >= group.get("max_delta", -np.inf):
            group["max_delta"] = delta
            group["max_linear"] = linear_index

    summaries: list[MismatchGroup] = []
    requests: list[ProbeRequest] = []
    for key in sorted(groups):
        group = groups[key]
        first_linear = int(group["first_linear"])
        first_coordinate = cast(
            "tuple[int, int, int]",
            tuple(
                int(value) for value in np.unravel_index(first_linear, mismatch.shape, order="F")
            ),
        )
        max_coordinate = cast(
            "tuple[int, int, int]",
            tuple(
                int(value)
                for value in np.unravel_index(group["max_linear"], mismatch.shape, order="F")
            ),
        )
        summary = MismatchGroup(
            matlab_scale=key[0],
            python_scale=key[1],
            boundary_class=key[2],
            mismatch_count=int(group["count"]),
            first_coordinate_zyx=first_coordinate,
            max_delta_coordinate_zyx=max_coordinate,
            max_abs_delta=float(group["max_delta"]),
        )
        summaries.append(summary)
        for reason, coordinate in (("first", first_coordinate), ("max_delta", max_coordinate)):
            requests.append(
                ProbeRequest(
                    request_id=f"g{len(summaries) - 1}_{reason}",
                    coordinate_zyx=coordinate,
                    reason=reason,
                    matlab_scale=key[0],
                    python_scale=key[1],
                    boundary_class=key[2],
                )
            )

    first_request = requests[0] if requests else None
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "provenance": provenance,
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "group_count": len(summaries),
        "groups": [asdict(summary) for summary in summaries],
        "probe_requests": [asdict(request) for request in requests],
        "first_request_id": first_request.request_id if first_request else None,
    }


def persist_energy_probe_payload(run_dir: Path, payload: dict[str, Any]) -> Path:
    """Write the versioned probe request contract beside proof artifacts."""
    path = run_dir / REQUEST_PATH
    write_json_with_hash(path, payload)
    write_text_with_hash(path.with_suffix(".txt"), _render_probe_payload(payload))
    return path


def record_hypothesis(
    run_dir: Path,
    *,
    proof_report: Path,
    first_failing_field: str,
    probe_request_id: str,
    hypothesis: str,
    expected_field: str,
    kind: str,
    design_review: bool = False,
) -> dict[str, Any]:
    """Append one isolated parity hypothesis and enforce the two-failure breaker."""
    path = run_dir / LEDGER_PATH
    records = _load_records(path)
    proof_fingerprint = payload_hash(json.loads(proof_report.read_text(encoding="utf-8")))
    matching_failures = [
        record
        for record in records
        if record["first_failing_field"] == first_failing_field
        and record["proof_fingerprint"] == proof_fingerprint
        and record["kind"] == "mathematical"
        and not record.get("design_review", False)
    ]
    if kind == "mathematical" and len(matching_failures) >= 2 and not design_review:
        raise RuntimeError(
            "Two failed mathematical hypotheses already target this proof field. "
            "Record a design review before another patch."
        )
    record = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "proof_report": str(proof_report),
        "proof_fingerprint": proof_fingerprint,
        "first_failing_field": first_failing_field,
        "probe_request_id": probe_request_id,
        "hypothesis": hypothesis,
        "expected_field": expected_field,
        "kind": kind,
        "design_review": design_review,
    }
    records.append(record)
    write_json_with_hash(path, {"schema_version": SCHEMA_VERSION, "records": records})
    return record


def ensure_rerun_allowed(run_dir: Path, *, stage: str) -> None:
    """Block a third Energy rerun after two unresolved mathematical hypotheses."""
    if stage != "energy":
        return
    records = _load_records(run_dir / LEDGER_PATH)
    by_field: dict[str, int] = defaultdict(int)
    for record in records:
        if record.get("kind") == "mathematical" and not record.get("design_review", False):
            by_field[str(record.get("first_failing_field", ""))] += 1
    blocked = sorted(field for field, count in by_field.items() if field and count >= 2)
    if blocked:
        raise RuntimeError(
            "Energy rerun blocked after two unresolved mathematical hypotheses for: "
            + ", ".join(blocked)
            + ". Record a design review before rerunning."
        )


def compare_probe_jsonl(matlab_path: Path, python_path: Path) -> dict[str, Any]:
    """Compare normalized MATLAB and Python probe records by request id."""
    matlab = _load_jsonl(matlab_path)
    python = _load_jsonl(python_path)
    ids = sorted(set(matlab) | set(python))
    differences: list[dict[str, Any]] = []
    for request_id in ids:
        if request_id not in matlab or request_id not in python:
            differences.append({"request_id": request_id, "kind": "missing_record"})
            continue
        mismatched_fields = _different_fields(matlab[request_id], python[request_id])
        if mismatched_fields:
            differences.append(
                {"request_id": request_id, "kind": "field_mismatch", "fields": mismatched_fields}
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "matlab_record_count": len(matlab),
        "python_record_count": len(python),
        "passed": not differences,
        "differences": differences,
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("records", [])) if isinstance(payload, dict) else []


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["request_id"])] = record
    return records


def _different_fields(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))


def _boundary_class(coordinate: tuple[int, int, int], shape: tuple[int, ...]) -> str:
    return (
        "boundary"
        if any(value in (0, size - 1) for value, size in zip(coordinate, shape))
        else "interior"
    )


def _require_same_shape(*arrays: np.ndarray) -> None:
    shapes = {np.asarray(array).shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Energy probe inputs must have one common shape, got {sorted(shapes)}")


def _render_probe_payload(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Adaptive Energy probe requests",
            f"Mismatches: {payload['mismatch_count']}",
            f"Groups: {payload['group_count']}",
            f"Requests: {len(payload['probe_requests'])}",
        ]
    )


__all__ = [
    "MismatchGroup",
    "ProbeRequest",
    "ProbeResult",
    "build_energy_probe_payload",
    "compare_probe_jsonl",
    "ensure_rerun_allowed",
    "persist_energy_probe_payload",
    "record_hypothesis",
]
