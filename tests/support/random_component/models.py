"""Typed models for the Random Component Parity structural gate.

These are the narrow, pure data types for the structural gate only.
Hessian / advisory diagnostics are intentionally excluded so the gate
has zero knowledge of energy samples, FFT work, or mode flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Mismatch:
    """A single structured difference (structural gate only)."""
    component: str
    path: str
    python: Any
    matlab: Any
    operands: dict[str, Any] | None = None
    ulp_distance: int | None = None


@dataclass(frozen=True)
class StructuralGateResult:
    """Result of a pure structural gate run.

    This is the primary, narrow return type for structural comparison.
    It contains only the information needed to decide "did the structural
    fields match?" and to produce the legacy report shape at the boundary.
    """
    passed: bool
    difference_count: int
    first_difference: dict[str, Any] | None = None
    linspace: dict[str, Any] = field(default_factory=dict)
    cases: list[dict[str, Any]] = field(default_factory=list)
    linspace_context_count: int = 0
    case_count: int = 0
    query_count_per_case: int = 16

    def to_report_dict(self) -> dict[str, Any]:
        """Produce the legacy structural_gate sub-object shape for report compat."""
        return {
            "passed": self.passed,
            "linspace_context_count": self.linspace_context_count,
            "case_count": self.case_count,
            "query_count_per_case": self.query_count_per_case,
        }
