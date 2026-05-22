"""Workflow helpers for SLAVV pipeline orchestration."""

from __future__ import annotations

from .pipeline import (
    PreparedPipelineRun,
    emit_progress,
    load_cached_stage_result,
    persist_stage_result,
    preprocess_image,
    resolve_edges_stage,
    resolve_energy_stage,
    resolve_network_stage,
    resolve_resumable_stage,
    resolve_stage_with_checkpoint,
    resolve_vertices_stage,
    stage_artifacts,
    validate_stage_control,
)
from .profiles import (
    PIPELINE_PROFILE_CHOICES,
    apply_pipeline_profile,
    get_pipeline_profile_defaults,
    normalize_pipeline_profile_name,
)

__all__ = [
    "PIPELINE_PROFILE_CHOICES",
    "PreparedPipelineRun",
    "apply_pipeline_profile",
    "emit_progress",
    "get_pipeline_profile_defaults",
    "load_cached_stage_result",
    "normalize_pipeline_profile_name",
    "persist_stage_result",
    "preprocess_image",
    "resolve_edges_stage",
    "resolve_energy_stage",
    "resolve_network_stage",
    "resolve_resumable_stage",
    "resolve_stage_with_checkpoint",
    "resolve_vertices_stage",
    "stage_artifacts",
    "validate_stage_control",
]
