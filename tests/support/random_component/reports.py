"""Report builders for Random Component Parity.

Produce the legacy dict shapes for JSON/CLI compatibility
while taking clean typed inputs (StructuralGateResult + hessian dicts).

This keeps the main module smaller and orchestration simple.
Use via the package __init__.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tests.support.random_component.models import StructuralGateResult


def build_structural_report(
    gate: StructuralGateResult, *, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build the legacy report shape from a pure StructuralGateResult.

    Used for --mode structural (fast/CI path). No hessian data.
    """
    first = gate.first_difference or {}
    return {
        "passed": gate.passed,
        "schema_version": 2,  # unchanged; internal refactor only (see hardening spec)
        "mode": "structural",
        "structural_gate": gate.to_report_dict(),
        "difference_count": gate.difference_count,
        "first_difference": first or None,
        "linspace": gate.linspace,
        "hessian_diagnostics": {
            "collected": False,
            "cases": [],
            "max_ulp_distance": 0,
            "worst_case_id": None,
            "worst_mismatch": None,
        },
        "cases": gate.cases,
        "differences": [],  # structural info is authoritative via gate
    }


def build_diagnostics_report(
    gate: StructuralGateResult,
    hess: dict[str, Any],
    *,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build full report for diagnostics using clean gate + hessian data."""
    hess_by_case = {h["case_id"]: h for h in hess.get("cases", [])}
    merged_cases: list[dict[str, Any]] = []
    for case in gate.cases:
        merged = dict(case)
        merged["hessian_diagnostics"] = hess_by_case.get(case["case_id"], {})
        merged_cases.append(merged)

    first = gate.first_difference or {}
    return {
        "passed": gate.passed,
        "schema_version": 2,  # unchanged; internal refactor only (see hardening spec)
        "mode": "diagnostics",
        "structural_gate": gate.to_report_dict(),
        "difference_count": gate.difference_count,
        "first_difference": first or None,
        "linspace": gate.linspace,
        "hessian_diagnostics": {
            "collected": True,
            "cases": hess.get("cases", []),
            "max_ulp_distance": hess.get("max_ulp_distance", 0),
            "worst_case_id": hess.get("worst_case_id"),
            "worst_mismatch": hess.get("worst_mismatch"),
        },
        "cases": merged_cases,
        "differences": [],
    }
