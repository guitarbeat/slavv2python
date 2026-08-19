"""Helpers for keeping curated app results internally consistent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slavv_python.schema.app_run import (
    AppRunState,
    counts_from_app_run,
    get_app_run,
    rebuild_network_for_curation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping


def summarize_processing_counts(
    processing_results: AppRunState | Mapping[str, Any],
) -> dict[str, int]:
    """Return the key count metrics used for curation comparisons."""
    return counts_from_app_run(AppRunState.from_value(processing_results))


def sync_curated_processing_results(
    processing_results: AppRunState | Mapping[str, Any],
    curated_vertices: dict[str, Any],
    curated_edges: dict[str, Any],
    *,
    baseline_counts: dict[str, int] | None = None,
) -> tuple[AppRunState, dict[str, int], dict[str, int]]:
    """Return updated processing results with a rebuilt network and stable baseline counts."""
    app_run = AppRunState.from_value(processing_results)
    preserved_baseline = (
        dict(baseline_counts)
        if baseline_counts is not None
        else summarize_processing_counts(app_run)
    )
    updated_run = rebuild_network_for_curation(app_run, curated_vertices, curated_edges)
    current_counts = summarize_processing_counts(updated_run)
    return updated_run, preserved_baseline, current_counts


def apply_curated_session_results(
    session_state: MutableMapping[str, Any],
    curated_vertices: dict[str, Any],
    curated_edges: dict[str, Any],
    *,
    curation_mode: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """Apply curated vertices and edges to session state and clear stale derived data."""
    updated_run, baseline_counts, current_counts = sync_curated_processing_results(
        get_app_run(session_state),
        curated_vertices,
        curated_edges,
        baseline_counts=session_state.get("curation_baseline_counts"),
    )
    session_state["processing_results"] = updated_run
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
