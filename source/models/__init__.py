"""Typed pipeline result models with dict compatibility helpers."""

from __future__ import annotations

from .results import (
    EdgeSet,
    EnergyResult,
    NetworkResult,
    PipelineResult,
    VertexSet,
    normalize_pipeline_result,
)

__all__ = [
    "EdgeSet",
    "EnergyResult",
    "NetworkResult",
    "PipelineResult",
    "VertexSet",
    "normalize_pipeline_result",
]
