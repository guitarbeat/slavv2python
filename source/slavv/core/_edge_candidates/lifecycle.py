"""Frontier lifecycle artifact helpers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def _normalize_manifest_candidate_index(value: Any) -> int | None:
    """Return a normalized manifest candidate index."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return int(value) if isinstance(value, str) and value != "" else None


def _update_origin_lifecycle_summary(
    summary: dict[str, Any],
    *,
    event: dict[str, Any],
    chosen_final_edge: bool,
) -> None:
    """Update one per-origin lifecycle summary row from a frontier event."""
    summary["terminal_hit_count"] += 1
    resolution_reason = str(event.get("resolution_reason", "unknown"))
    resolution_counts = cast("dict[str, int]", summary["resolution_counts"])
    resolution_counts[resolution_reason] = int(resolution_counts.get(resolution_reason, 0)) + 1
    if event["claim_reassigned"]:
        summary["claim_reassignment_count"] += 1
        reassignment_reason = event.get("claim_reassignment_reason")
        if (
            isinstance(reassignment_reason, str)
            and reassignment_reason
            and reassignment_reason not in summary["claim_reassignment_samples"]
            and len(summary["claim_reassignment_samples"]) < 3
        ):
            summary["claim_reassignment_samples"].append(reassignment_reason)
    if event["survived_candidate_manifest"]:
        summary["emitted_candidate_count"] += 1
        if chosen_final_edge:
            summary["chosen_final_edge_count"] += 1
        else:
            summary["final_cleanup_loss_count"] += 1
        endpoint_pair = event.get("emitted_endpoint_pair")
        if (
            isinstance(endpoint_pair, list)
            and endpoint_pair not in summary["emitted_endpoint_pair_samples"]
            and len(summary["emitted_endpoint_pair_samples"]) < 3
        ):
            summary["emitted_endpoint_pair_samples"].append(endpoint_pair)
        return

    summary["rejected_terminal_count"] += 1
    rejection_reason = event.get("rejection_reason")
    if (
        isinstance(rejection_reason, str)
        and rejection_reason
        and rejection_reason not in summary["rejection_reason_samples"]
        and len(summary["rejection_reason_samples"]) < 3
    ):
        summary["rejection_reason_samples"].append(rejection_reason)


def _build_frontier_candidate_lifecycle(
    candidates: dict[str, Any],
    chosen_candidate_indices: np.ndarray | list[int] | None = None,
) -> dict[str, Any]:
    """Build a JSON-friendly frontier lifecycle artifact for shared-neighborhood audits."""
    raw_events = candidates.get("frontier_lifecycle_events", [])
    chosen_indices = {
        int(index)
        for index in np.asarray(
            chosen_candidate_indices if chosen_candidate_indices is not None else [], dtype=np.int32
        ).reshape(-1)
    }
    events: list[dict[str, Any]] = []
    per_origin_summary: dict[int, dict[str, Any]] = {}

    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        event = dict(raw_event)
        seed_origin_index = int(event.get("seed_origin_index", -1))
        manifest_candidate_index = _normalize_manifest_candidate_index(
            event.get("manifest_candidate_index")
        )
        chosen_final_edge = (
            manifest_candidate_index is not None and manifest_candidate_index in chosen_indices
        )
        event["seed_origin_index"] = seed_origin_index
        event["terminal_vertex_index"] = int(event.get("terminal_vertex_index", -1))
        event["terminal_hit_sequence"] = int(event.get("terminal_hit_sequence", 0))
        event["survived_candidate_manifest"] = bool(event.get("survived_candidate_manifest", False))
        event["manifest_candidate_index"] = manifest_candidate_index
        event["chosen_final_edge"] = chosen_final_edge
        event["claim_reassigned"] = bool(event.get("claim_reassigned", False))
        if not event["claim_reassigned"]:
            event["claim_reassignment_reason"] = None
        elif not isinstance(event.get("claim_reassignment_reason"), str):
            event["claim_reassignment_reason"] = "reassigned_from_parent_child_resolution"
        event["final_survival_stage"] = (
            "final_edge_retained"
            if chosen_final_edge
            else (
                "final_cleanup_dropped"
                if event["survived_candidate_manifest"]
                else "pre_manifest_rejection"
            )
        )
        events.append(event)

        summary = per_origin_summary.setdefault(
            seed_origin_index,
            {
                "origin_index": seed_origin_index,
                "terminal_hit_count": 0,
                "emitted_candidate_count": 0,
                "rejected_terminal_count": 0,
                "chosen_final_edge_count": 0,
                "claim_reassignment_count": 0,
                "final_cleanup_loss_count": 0,
                "resolution_counts": {},
                "emitted_endpoint_pair_samples": [],
                "rejection_reason_samples": [],
                "claim_reassignment_samples": [],
            },
        )
        _update_origin_lifecycle_summary(
            summary,
            event=event,
            chosen_final_edge=chosen_final_edge,
        )

    per_origin_payload = [
        per_origin_summary[origin_index] for origin_index in sorted(per_origin_summary)
    ]
    per_origin_payload.sort(
        key=lambda item: (
            -int(item["terminal_hit_count"]),
            -int(item["rejected_terminal_count"]),
            int(item["origin_index"]),
        )
    )
    return {
        "schema_version": 2,
        "frontier_terminal_hit_event_count": len(events),
        "frontier_terminal_accept_event_count": len(
            [event for event in events if event.get("survived_candidate_manifest")]
        ),
        "frontier_terminal_reject_event_count": len(
            [event for event in events if not event.get("survived_candidate_manifest")]
        ),
        "events": events,
        "per_origin_summary": per_origin_payload,
        "top_origin_summaries": per_origin_payload[:5],
    }
