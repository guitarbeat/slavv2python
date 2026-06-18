"""Execution policies for Energy stage computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


from slavv_python.pipeline.policy import PipelinePolicy


@dataclass(frozen=True)
class EnergyPolicy:
    """Encapsulates the structural differences between execution modes."""
    
    precision: np.dtype
    intensity_scaling: bool
    downsample_alignment: str  # "paper" | "matlab"
    mesh_strategy: str         # "arithmetic" | "linspace"
    interpolation_mode: str    # "standard" | "matlab_inf_prop"
    exact_sign_clipping: bool
    
    @classmethod
    def from_params(cls, params: dict[str, Any]) -> EnergyPolicy:
        """Create a policy from pipeline parameters."""
        pipeline = PipelinePolicy.from_params(params)
        is_exact = pipeline.internal_grid_alignment == "matlab"
        
        if is_exact:
            return cls(
                precision=pipeline.precision,
                intensity_scaling=False,
                downsample_alignment="matlab",
                mesh_strategy="linspace",
                interpolation_mode="matlab_inf_prop",
                exact_sign_clipping=True,
            )
        
        return cls(
            precision=pipeline.precision,
            intensity_scaling=True,
            downsample_alignment="paper",
            mesh_strategy="arithmetic",
            interpolation_mode="standard",
            exact_sign_clipping=False,
        )


__all__ = ["EnergyPolicy"]
