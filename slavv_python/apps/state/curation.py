"""Helpers for keeping curated app results internally consistent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slavv_python.core import SlavvPipeline
from slavv_python.models import normalize_pipeline_result

if TYPE_CHECKING:
    from collections.abc import MutableMapping


def summarize_processing_counts(processing_results: dict[str, Any]) -> dict[str, int]:
    """Return the key count metrics used for curation comparisons."""
    typed_result = normalize_pipeline_result(processing_results)
    results = typed_result.to_dict()
    network = results.get("network", {})
    return {
        "Vertices": len(results.get("vertices", {}).get("positions", [])),
        "Edges": len(results.get("edges", {}).get("traces", [])),
        "Strands": len(network.get("strands", [])),
        "Bifurcations": len(network.get("bifurcations", [])),
    }


def sync_curated_processing_results(
    processing_results: dict[str, Any],
    curated_vertices: dict[str, Any],
    curated_edges: dict[str, Any],
    *,
    baseline_counts: dict[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    """Return updated processing results with a rebuilt network and stable baseline counts."""
    typed_result = normalize_pipeline_result(processing_results)
    preserved_baseline = (
        dict(baseline_counts)
        if baseline_counts is not None
        else summarize_processing_counts(processing_results)
    )

    rebuilt_network = SlavvPipeline().build_network(
        curated_edges,
        curated_vertices,
        typed_result.parameters,
    )
    updated_results = typed_result.to_dict()
    updated_results["vertices"] = curated_vertices
    updated_results["edges"] = curated_edges
    updated_results["network"] = rebuilt_network

    return updated_results, preserved_baseline, summarize_processing_counts(updated_results)


def apply_curated_session_results(
    session_state: MutableMapping[str, Any],
    curated_vertices: dict[str, Any],
    curated_edges: dict[str, Any],
    *,
    curation_mode: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """Apply curated vertices and edges to session state and clear stale derived data."""
    updated_results, baseline_counts, current_counts = sync_curated_processing_results(
        session_state["processing_results"],
        curated_vertices,
        curated_edges,
        baseline_counts=session_state.get("curation_baseline_counts"),
    )
    session_state["processing_results"] = updated_results
    session_state["curation_baseline_counts"] = baseline_counts
    session_state["last_curation_mode"] = curation_mode
    session_state.pop("share_report_prepared_signature", None)
    return baseline_counts, current_counts


def build_curation_stats_rows(
    baseline_counts: dict[str, int],
    current_counts: dict[str, int],
) -> list[dict[str, str | int]]:
    """Build curation comparison rows with signed deltas."""
    rows: list[dict[str, str | int]] = []
    for component in ("Vertices", "Edges", "Strands", "Bifurcations"):
        original = int(baseline_counts.get(component, 0))
        current = int(current_counts.get(component, 0))
        delta = current - original
        change_pct = (
            "n/a" if original == 0 and current != 0 else f"{((delta / original) * 100.0):+.2f}"
        )
        if original == 0 and current == 0:
            change_pct = "+0.00"
        rows.append(
            {
                "Component": component,
                "Original": original,
                "Current": current,
                "Delta": delta,
                "Change (%)": change_pct,
            }
        )
    return rows


__all__ = [
    "apply_curated_session_results",
    "build_curation_stats_rows",
    "summarize_processing_counts",
    "sync_curated_processing_results",
]
