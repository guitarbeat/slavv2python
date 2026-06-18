"""Global execution policies for the SLAVV pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PipelinePolicy:
    """Encapsulates the structural and mathematical differences between paths."""
    
    precision: np.dtype
    internal_grid_alignment: str  # "matlab" | "paper"
    rounding_mode: str            # "half-up" | "even"
    energy_engine: str            # "incremental" | "parallel"

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> PipelinePolicy:
        """Create a policy from pipeline parameters."""
        is_exact = bool(params.get("comparison_exact_network", False))
        
        if is_exact:
            # Exact Route (Innovation Path)
            return cls(
                precision=np.dtype(np.float64),
                internal_grid_alignment="matlab",
                rounding_mode="half-up",
                energy_engine="incremental",
            )
        
        # Paper Path (Baseline Path)
        return cls(
            precision=np.dtype(np.float32),
            internal_grid_alignment="paper",
            rounding_mode="even",
            energy_engine="parallel",
        )

    def round(self, values: np.ndarray) -> np.ndarray:
        """Apply policy-specific rounding."""
        if self.rounding_mode == "half-up":
            return np.floor(np.asarray(values, dtype=float) + 0.5)
        return np.round(values)


__all__ = ["PipelinePolicy"]
