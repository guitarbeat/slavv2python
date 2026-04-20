"""Helpers for normalized visualization-page state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slavv.models import normalize_pipeline_result

if TYPE_CHECKING:
    from collections.abc import Mapping


def normalize_visualization_results(processing_results: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized dict payload for visualization consumers."""
    return normalize_pipeline_result(processing_results).to_dict()


def list_available_visualizations(processing_results: Mapping[str, Any]) -> list[str]:
    """Return visualization modes supported by the current payload."""
    typed_result = normalize_pipeline_result(processing_results)
    available: list[str] = []

    if typed_result.energy_data is not None:
        available.append("Energy Field")
    if (
        typed_result.vertices is not None
        and typed_result.edges is not None
        and typed_result.network is not None
    ):
        available.extend(["2D Network", "3D Network", "Depth Projection", "Strand Analysis"])

    return available


__all__ = ["list_available_visualizations", "normalize_visualization_results"]
