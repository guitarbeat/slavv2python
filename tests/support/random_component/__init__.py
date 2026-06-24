"""Random Component Parity support package.

Contains narrow, typed models and the pure structural gate.
The goal is a clean separation so the structural gate never sees
Hessian, energy samples, or advisory collection.
"""
from .diagnostics import collect_hessian_diagnostics
from .gate import run_structural_gate
from .models import Mismatch, StructuralGateResult

__all__ = [
    "Mismatch",
    "StructuralGateResult",
    "collect_hessian_diagnostics",
    "run_structural_gate",
]
