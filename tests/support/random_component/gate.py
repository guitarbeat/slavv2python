"""Pure structural gate for the Random Component Parity suite.

This module contains the narrow structural-only comparison path.
It must never import or call anything related to Hessian, energy samples,
curvatures, gradients, or advisory diagnostics.
"""

from __future__ import annotations

from typing import Any

from tests.support.random_component.models import (
    StructuralGateResult,
)


# Low-level structural comparators are imported inside the function to avoid
# circular import (main module imports this gate module).
# In later phases the compare helpers will live alongside the gate.
def _get_structural_comparators():
    from tests.support.random_component_parity import (
        QUERY_COUNT_PER_CASE,
        _compare_case_section,
        _compare_linspace_section,
    )
    return QUERY_COUNT_PER_CASE, _compare_case_section, _compare_linspace_section


def run_structural_gate(
    python: dict[str, Any],
    matlab: dict[str, Any],
    *,
    manifest: dict[str, Any] | None = None,
) -> StructuralGateResult:
    """Run only the structural gate.

    Returns a narrow StructuralGateResult containing only linspace,
    interp3, padded_shape, coordinates, and valid results.
    Never touches energy samples or produces hessian diagnostics.
    """
    QUERY_COUNT_PER_CASE, _compare_case_section, _compare_linspace_section = (
        _get_structural_comparators()
    )

    seed_by_case = {
        str(case["id"]): int(case["seed"])
        for case in (manifest or {}).get("cases", [])
    }

    linspace_differences_raw = _compare_linspace_section(python, matlab)
    case_differences_raw: list[dict[str, Any]] = []
    case_reports: list[dict[str, Any]] = []

    if len(python.get("cases", [])) != len(matlab.get("cases", [])):
        case_differences_raw.append(
            {
                "path": "cases",
                "python_size": len(python.get("cases", [])),
                "matlab_size": len(matlab.get("cases", [])),
                "component": "case_count",
            }
        )

    for index, record in enumerate(matlab.get("cases", [])):
        if index >= len(python.get("cases", [])):
            break
        python_case = python["cases"][index]
        case_id = python_case["case_id"]
        seed = seed_by_case.get(case_id, -1)

        differences = _compare_case_section(python_case, record, seed=seed)
        case_reports.append(
            {
                "case_id": case_id,
                "seed": seed,
                "passed": not differences,
                "difference_count": len(differences),
                "first_difference": differences[0] if differences else None,
            }
        )
        case_differences_raw.extend(differences)

    all_differences = linspace_differences_raw + case_differences_raw
    passed = len(all_differences) == 0

    first_diff = all_differences[0] if all_differences else None

    linspace_section = {
        "passed": len(linspace_differences_raw) == 0,
        "difference_count": len(linspace_differences_raw),
        "first_difference": linspace_differences_raw[0] if linspace_differences_raw else None,
    }

    return StructuralGateResult(
        passed=passed,
        difference_count=len(all_differences),
        first_difference=first_diff,
        linspace=linspace_section,
        cases=case_reports,
        linspace_context_count=len(python.get("linspace", [])),
        case_count=len(python.get("cases", [])),
        query_count_per_case=QUERY_COUNT_PER_CASE,
    )
