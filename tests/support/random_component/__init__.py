"""Random Component Parity support package.

Contains narrow, typed models and the pure structural gate.
The goal is a clean separation so the structural gate never sees
Hessian, energy samples, or advisory collection.
"""
from .gate import run_structural_gate
from .models import Mismatch, StructuralGateResult

__all__ = ["Mismatch", "StructuralGateResult", "run_structural_gate"]
