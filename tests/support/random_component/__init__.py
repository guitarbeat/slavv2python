"""Random Component Parity support package.

Public API for the hardened suite (post Phase 1-3 refactor):

- run_structural_gate: pure structural comparison (linspace, interp3, padded_shape,
  coordinates, valid). Never touches Hessian.
- collect_hessian_diagnostics: advisory ULP collection (diagnostics mode only).
- build_structural_report / build_diagnostics_report: produce legacy report dicts.
- StructuralGateResult, Mismatch: typed results.

Internal modules:
- models.py: dataclasses
- gate.py: structural only
- diagnostics.py: hessian only
- reports.py: builders

See the hardening spec and PARITY_RANDOM_COMPONENT_SUITE.md for details.
The goal is maintainability, small files, and clear separation while preserving
exact structural gate behavior vs MATLAB baseline.
"""

from .diagnostics import collect_hessian_diagnostics
from .gate import run_structural_gate
from .models import Mismatch, StructuralGateResult
from .reports import build_diagnostics_report, build_structural_report

__all__ = [
    "Mismatch",
    "StructuralGateResult",
    "build_diagnostics_report",
    "build_structural_report",
    "collect_hessian_diagnostics",
    "run_structural_gate",
]
