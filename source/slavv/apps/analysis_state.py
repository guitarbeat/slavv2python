"""Helpers for normalized analysis-page state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slavv.apps.curation_state import summarize_processing_counts
from slavv.models import normalize_pipeline_result

if TYPE_CHECKING:
    from collections.abc import Mapping


def normalize_analysis_results(processing_results: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized dict payload for analysis consumers."""
    return normalize_pipeline_result(processing_results).to_dict()


def has_analysis_network(processing_results: Mapping[str, Any]) -> bool:
    """Return whether analysis can proceed on the provided results."""
    return normalize_pipeline_result(processing_results).network is not None


def resolve_analysis_stats(
    processing_results: Mapping[str, Any],
    analysis_stats: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return existing analysis stats or derive baseline counts when absent."""
    if analysis_stats is not None:
        return dict(analysis_stats)
    return summarize_processing_counts(normalize_analysis_results(processing_results))


__all__ = [
    "has_analysis_network",
    "normalize_analysis_results",
    "resolve_analysis_stats",
]
