"""Helpers for keeping curated app results internally consistent."""

from __future__ import annotations

from typing import Any

from slavv.core import SLAVVProcessor


def summarize_processing_counts(processing_results: dict[str, Any]) -> dict[str, int]:
    """Return the key count metrics used for curation comparisons."""
    network = processing_results.get("network", {})
    return {
        "Vertices": len(processing_results.get("vertices", {}).get("positions", [])),
        "Edges": len(processing_results.get("edges", {}).get("traces", [])),
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
    preserved_baseline = (
        dict(baseline_counts)
        if baseline_counts is not None
        else summarize_processing_counts(processing_results)
    )

    updated_results = dict(processing_results)
    updated_results["vertices"] = curated_vertices
    updated_results["edges"] = curated_edges
    updated_results["network"] = SLAVVProcessor().construct_network(
        curated_edges,
        curated_vertices,
        updated_results.get("parameters", {}),
    )

    return updated_results, preserved_baseline, summarize_processing_counts(updated_results)


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
    "build_curation_stats_rows",
    "summarize_processing_counts",
    "sync_curated_processing_results",
]
