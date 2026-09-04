"""Durable timing payloads for the Phase 2 Edges profile.

Observational only: this records existing discovery and selection spans without
changing either algorithm or array layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from slavv_python.analytics.parity.utils import now_iso, write_json_with_hash

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EdgeTimingRecord:
    """Durable observational record for one Edges execution.

    The record deliberately models discovery and selection as separate spans;
    callers can persist it without exposing algorithm internals or changing
    checkpoint formats.
    """

    discovery_seconds: float
    selection_seconds: float
    candidate_count: int
    edge_count: int
    exact_route: bool
    writer_authorized: bool
    started_at: str
    completed_at: str

    def to_payload(self) -> dict[str, Any]:
        return build_edge_timing_payload(
            discovery_seconds=self.discovery_seconds,
            selection_seconds=self.selection_seconds,
            candidate_count=self.candidate_count,
            edge_count=self.edge_count,
            exact_route=self.exact_route,
            writer_authorized=self.writer_authorized,
            started_at=self.started_at,
            completed_at=self.completed_at,
        )


def build_edge_timing_payload(
    *,
    discovery_seconds: float,
    selection_seconds: float,
    candidate_count: int,
    edge_count: int,
    exact_route: bool,
    writer_authorized: bool,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Build the stable, JSON-safe timing contract for an Edges execution."""
    discovery = max(0.0, float(discovery_seconds))
    selection = max(0.0, float(selection_seconds))
    discovery_span_key = (
        "watershed_discovery_seconds" if exact_route else "tracing_discovery_seconds"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "edges",
        "profile": "exact-route" if exact_route else "paper",
        "exact_route": bool(exact_route),
        "writer_authorized": bool(writer_authorized),
        "discovery_strategy": "watershed" if exact_route else "tracing",
        "precision": "float64" if exact_route else "float32",
        "started_at": started_at or now_iso(),
        "completed_at": completed_at or now_iso(),
        "candidate_count": int(candidate_count),
        "edge_count": int(edge_count),
        "discovery_seconds": discovery,
        "selection_seconds": selection,
        "total_seconds": discovery + selection,
        "spans": {
            discovery_span_key: discovery,
            "edge_selection_seconds": selection,
        },
    }


def write_edge_timing(path: Path, payload: dict[str, Any]) -> Path:
    """Persist timing JSON and a physical-file SHA-256 sidecar."""
    return write_json_with_hash(path, payload)


__all__ = [
    "SCHEMA_VERSION",
    "EdgeTimingRecord",
    "build_edge_timing_payload",
    "write_edge_timing",
]
