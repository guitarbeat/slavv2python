"""Helpers for loading dashboard context from session-backed state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast

from source.apps._state_utils import normalize_state_results
from source.apps.services.exports import has_full_network_results
from source.apps.share_report import compute_shareable_stats
from source.runtime import RunSnapshot, load_run_snapshot

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class DashboardContext(TypedDict):
    """Typed session-backed inputs for the dashboard surface."""

    run_dir: str | None
    snapshot: RunSnapshot | None
    results: dict[str, Any] | None
    share_metrics: dict[str, Any]
    dataset_name: str
    stats: dict[str, Any] | None


def normalize_dashboard_results(processing_results: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized dict payload for dashboard consumers."""
    return normalize_state_results(processing_results)


def resolve_dashboard_stats(
    results: Mapping[str, Any] | None,
    *,
    image_shape: tuple[int, int, int],
    stats_builder: Callable[..., dict[str, Any]] = compute_shareable_stats,
) -> dict[str, Any] | None:
    """Compute dashboard statistics when a full network result is available."""
    if not results or not has_full_network_results(results):
        return None
    return stats_builder(results, image_shape=image_shape)


def load_dashboard_context(
    session_state: Mapping[str, object],
    *,
    snapshot_loader: Callable[[str], RunSnapshot | None] = load_run_snapshot,
    stats_builder: Callable[..., dict[str, Any]] = compute_shareable_stats,
) -> DashboardContext:
    """Load dashboard context from session state and run metadata."""
    run_dir = cast("str | None", session_state.get("current_run_dir"))
    snapshot = snapshot_loader(run_dir) if run_dir else None
    raw_results = cast("dict[str, Any] | None", session_state.get("processing_results"))
    results = None if raw_results is None else normalize_dashboard_results(raw_results)
    share_metrics = cast("dict[str, Any]", session_state.get("share_report_metrics", {}))
    dataset_name = str(session_state.get("dataset_name", "No dataset loaded"))
    image_shape = cast("tuple[int, int, int]", session_state.get("image_shape", (100, 100, 50)))
    stats = resolve_dashboard_stats(results, image_shape=image_shape, stats_builder=stats_builder)

    return {
        "run_dir": run_dir,
        "snapshot": snapshot,
        "results": results,
        "share_metrics": share_metrics,
        "dataset_name": dataset_name,
        "stats": stats,
    }


__all__ = [
    "DashboardContext",
    "load_dashboard_context",
    "normalize_dashboard_results",
    "resolve_dashboard_stats",
]
