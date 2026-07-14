"""Compare MATLAB and Python watershed strel-state JSONL traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_states(path: Path) -> dict[int, dict[str, Any]]:
    states: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "strel_state":
            states[int(event["iteration"])] = event
    return states


def _matlab_to_python_linear(value: Any) -> int:
    return round(float(value)) - 1


def _as_float(value: Any) -> float:
    return float(value)


def _normalize_matlab_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["linear"] = _matlab_to_python_linear(row["linear"])
    normalized["strel_index"] = round(float(row["strel_index"])) - 1
    return normalized


def _normalize_python_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["linear"] = int(row["linear"])
    normalized["strel_index"] = int(row["strel_index"])
    return normalized


def _row_key(row: dict[str, Any]) -> int:
    return int(row["linear"])


def _row_delta(
    matlab_row: dict[str, Any],
    python_row: dict[str, Any],
    *,
    atol: float,
) -> dict[str, Any]:
    fields = (
        "adjusted_energy",
        "vertex_index_before_claim",
        "pointer_before_claim",
        "d_over_r_before_claim",
        "size_before_claim",
        "vertex_index_after_claim",
        "pointer_after_claim",
        "d_over_r_after_claim",
        "size_after_claim",
    )
    mismatch: dict[str, Any] = {}
    for field in fields:
        left = matlab_row.get(field)
        right = python_row.get(field)
        if isinstance(left, bool) or isinstance(right, bool):
            equal = bool(left) == bool(right)
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            equal = bool(np.isclose(_as_float(left), _as_float(right), rtol=0.0, atol=atol))
        else:
            equal = left == right
        if not equal:
            mismatch[field] = {"matlab": left, "python": right}
    return mismatch


def _compare_iteration(
    iteration: int,
    matlab_state: dict[str, Any],
    python_state: dict[str, Any] | None,
    *,
    atol: float,
) -> dict[str, Any]:
    matlab_current = _matlab_to_python_linear(matlab_state["current_linear"])
    result: dict[str, Any] = {
        "iteration": iteration,
        "matlab_current_linear_0based": matlab_current,
        "matlab_current_vertex_index": round(float(matlab_state["current_vertex_index"])),
    }
    if python_state is None:
        result["status"] = "missing_python_state"
        return result

    python_current = int(python_state["current_linear"])
    result.update(
        {
            "python_current_linear_0based": python_current,
            "python_current_vertex_index": int(python_state["current_vertex_index"]),
        }
    )
    if matlab_current != python_current:
        result["status"] = "different_current_location"
        return result

    matlab_rows = {
        _row_key(_normalize_matlab_row(row)): _normalize_matlab_row(row)
        for row in matlab_state.get("top_adjusted", [])
    }
    python_rows = {
        _row_key(_normalize_python_row(row)): _normalize_python_row(row)
        for row in python_state.get("top_adjusted", [])
    }
    shared = sorted(set(matlab_rows) & set(python_rows))
    mismatches = [
        {
            "linear": linear,
            "mismatch": _row_delta(matlab_rows[linear], python_rows[linear], atol=atol),
        }
        for linear in shared
    ]
    mismatches = [item for item in mismatches if item["mismatch"]]
    result.update(
        {
            "status": "match" if not mismatches else "row_mismatch",
            "shared_top_adjusted_rows": len(shared),
            "matlab_only_top_rows": sorted(set(matlab_rows) - set(python_rows))[:10],
            "python_only_top_rows": sorted(set(python_rows) - set(matlab_rows))[:10],
            "mismatches": mismatches[:10],
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab-trace", type=Path, required=True)
    parser.add_argument("--python-trace", type=Path, required=True)
    parser.add_argument("--iteration", type=int, action="append", default=[])
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args(argv)

    matlab_states = _load_states(args.matlab_trace)
    python_states = _load_states(args.python_trace)
    iterations = args.iteration or sorted(set(matlab_states) | set(python_states))
    report = [
        _compare_iteration(
            iteration,
            matlab_states[iteration],
            python_states.get(iteration),
            atol=args.atol,
        )
        for iteration in iterations
        if iteration in matlab_states
    ]
    print(json.dumps(report, indent=2))
    return 0 if all(item["status"] == "match" for item in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
