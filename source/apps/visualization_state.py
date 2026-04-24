"""Helpers for normalized visualization-page state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

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


def has_visualization_network(processing_results: Mapping[str, Any]) -> bool:
    """Return whether the payload contains the full network needed for exports."""
    typed_result = normalize_pipeline_result(processing_results)
    return (
        typed_result.vertices is not None
        and typed_result.edges is not None
        and typed_result.network is not None
    )


def extract_visualization_export_payload(
    processing_results: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return normalized vertices, edges, network, and parameters for export consumers."""
    normalized = normalize_visualization_results(processing_results)
    return (
        cast("dict[str, Any]", normalized["vertices"]),
        cast("dict[str, Any]", normalized["edges"]),
        cast("dict[str, Any]", normalized["network"]),
        cast("dict[str, Any]", normalized["parameters"]),
    )


def resolve_visualization_session_context(
    session_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Return session-derived context used by visualization exports and sharing."""
    return {
        "run_dir": cast("str | None", session_state.get("current_run_dir")),
        "dataset_name": cast("str", session_state.get("dataset_name", "SLAVV dataset")),
        "image_shape": cast(
            "tuple[int, int, int]", session_state.get("image_shape", (100, 100, 50))
        ),
        "share_metrics": cast("dict[str, int]", session_state.get("share_report_metrics", {})),
    }


__all__ = [
    "extract_visualization_export_payload",
    "has_visualization_network",
    "list_available_visualizations",
    "normalize_visualization_results",
    "resolve_visualization_session_context",
]
