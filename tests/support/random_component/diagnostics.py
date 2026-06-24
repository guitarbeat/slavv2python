"""Hessian / advisory diagnostics collection.

Completely separate from the structural gate.
Only invoked for --mode diagnostics.

Public entry: collect_hessian_diagnostics()
"""

from __future__ import annotations

from typing import Any

from tests.support.random_component_parity import (
    _as_list,
    _compare_values,
)


def collect_hessian_diagnostics(
    python: dict[str, Any],
    matlab: dict[str, Any],
) -> dict[str, Any]:
    """Collect Hessian ULP diagnostics only.

    Returns a dict suitable for the hessian_diagnostics section of the report.
    Does not perform any structural comparisons.
    """
    hessian_cases: list[dict[str, Any]] = []
    worst: dict[str, Any] | None = None
    max_ulp = 0

    py_cases = python.get("cases", [])
    mat_cases = matlab.get("cases", [])
    for py_case, mat_case in zip(
        py_cases, mat_cases
    ):  # tolerate length mismatch (structural gate reports it)
        case_id = py_case["case_id"]
        hess = _hessian_for_case(py_case, mat_case)
        hessian_cases.append({"case_id": case_id, **hess})

        if hess["max_ulp_distance"] > max_ulp:
            max_ulp = hess["max_ulp_distance"]
            worst = hess.get("worst_mismatch")

    return {
        "collected": True,
        "cases": hessian_cases,
        "max_ulp_distance": max_ulp,
        "worst_case_id": worst.get("case_id") if worst else None,
        "worst_mismatch": worst,
    }


def _hessian_for_case(
    python_case: dict[str, Any],
    matlab_record: Any,
) -> dict[str, Any]:
    """Internal: summarize per-case Hessian drift."""
    matlab_samples = _as_list(matlab_record.samples)
    python_samples = python_case["energy"]["samples"]
    worst: dict[str, Any] | None = None
    mismatch_count = 0

    for sample_index, (python_sample, matlab_sample) in enumerate(
        zip(python_samples, matlab_samples)
    ):
        coordinate = [int(value) for value in python_sample["coordinate_yxz"]]
        for field in ("curvatures", "gradient", "laplacian", "energy"):
            python_value = python_sample[field]
            matlab_value = getattr(matlab_sample, field)
            if field in ("curvatures", "gradient"):
                for value_index, (left, right) in enumerate(
                    zip(python_value, _as_list(matlab_value), strict=True)
                ):
                    diff = _compare_values(
                        f"cases[{python_case['case_id']}].energy.samples[{sample_index}].{field}[{value_index}]",
                        left,
                        float(right),
                    )
                    if diff is None:
                        continue
                    mismatch_count += 1
                    if worst is None or diff.get("ulp_distance", 0) > (
                        worst.get("ulp_distance") or 0
                    ):
                        worst = {
                            **diff,
                            "component": f"energy.{field}",
                            "coordinate_yxz": coordinate,
                            "case_id": python_case["case_id"],
                        }
            else:
                diff = _compare_values(
                    f"cases[{python_case['case_id']}].energy.samples[{sample_index}].{field}",
                    python_value,
                    float(matlab_value),
                )
                if diff is None:
                    continue
                mismatch_count += 1
                if worst is None or diff.get("ulp_distance", 0) > (worst.get("ulp_distance") or 0):
                    worst = {
                        **diff,
                        "component": f"energy.{field}",
                        "coordinate_yxz": coordinate,
                        "case_id": python_case["case_id"],
                    }

    return {
        "mismatch_count": mismatch_count,
        "max_ulp_distance": 0 if worst is None else int(worst["ulp_distance"]),
        "worst_mismatch": worst,
    }
