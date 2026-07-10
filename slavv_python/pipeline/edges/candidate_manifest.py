"""CandidateManifest: typed Edge Discovery candidate Stage Result.

Deep module for the candidate payload contract used between discovery,
selection, audit, and resumable unit aggregation. Legacy dict shells are
adapted here so callers do not re-implement append/reorder/endpoint logic.

Wire-format field names (including connection_sources values such as
``\"frontier\"``) stay stable for checkpoints and audits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from slavv_python.pipeline.edges.payloads import _merge_edge_diagnostics

_ALLOWED_CONNECTION_SOURCES = frozenset(
    {"frontier", "watershed", "geodesic", "fallback", "unknown"}
)


def normalize_candidate_connection_sources(
    raw_sources: Any,
    candidate_connection_count: int,
    *,
    default_source: str = "unknown",
) -> list[str]:
    """Return a normalized per-connection source label list."""
    if candidate_connection_count <= 0:
        return []

    if isinstance(raw_sources, np.ndarray):
        source_values = np.asarray(raw_sources).reshape(-1).tolist()
    elif isinstance(raw_sources, (list, tuple)):
        source_values = list(raw_sources)
    else:
        source_values = []

    default_label = default_source if default_source in _ALLOWED_CONNECTION_SOURCES else "unknown"
    normalized: list[str] = []
    for index in range(candidate_connection_count):
        if index < len(source_values):
            source_label = str(source_values[index]).strip().lower()
            normalized.append(
                source_label if source_label in _ALLOWED_CONNECTION_SOURCES else default_label
            )
            continue
        normalized.append(default_label)
    return normalized


def candidate_as_payload(candidates: CandidateManifest | dict[str, Any]) -> dict[str, Any]:
    """Return a legacy dict payload, flattening typed manifests at explicit seams."""
    if isinstance(candidates, CandidateManifest):
        return candidates.to_payload()
    return candidates


def endpoint_pairs_from_connections(connections: np.ndarray) -> set[tuple[int, int]]:
    """Orientation-independent endpoint pairs from a connections array."""
    return CandidateManifest(
        connections=np.asarray(connections, dtype=np.int32).reshape(-1, 2),
    ).endpoint_pair_set()


def incident_pair_counts(connections: np.ndarray) -> dict[int, int]:
    """Count unique incident endpoint pairs for each vertex."""
    counts: dict[int, int] = {}
    for start_vertex, end_vertex in endpoint_pairs_from_connections(connections):
        counts[int(start_vertex)] = counts.get(int(start_vertex), 0) + 1
        counts[int(end_vertex)] = counts.get(int(end_vertex), 0) + 1
    return counts


def reorder_candidate_payload(
    candidates: dict[str, Any],
    sort_order: np.ndarray,
) -> dict[str, Any]:
    """Return a new candidate dict reordered by the provided sort indices."""
    return CandidateManifest.from_payload(candidates).reordered(sort_order).to_payload()


@dataclass
class CandidateManifest:
    """Typed edge-candidate payload produced by an Edge Discovery strategy."""

    traces: list[np.ndarray] = field(default_factory=list)
    connections: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int32))
    metrics: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    energy_traces: list[np.ndarray] = field(default_factory=list)
    scale_traces: list[np.ndarray] = field(default_factory=list)
    origin_indices: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    connection_sources: list[str] = field(default_factory=list)
    frontier_lifecycle_events: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> CandidateManifest:
        """Return an empty candidate manifest."""
        return cls()

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CandidateManifest:
        """Build a typed manifest from a legacy candidate dict shell."""
        payload_copy = dict(payload)
        traces = [np.asarray(trace, dtype=np.float64) for trace in payload_copy.pop("traces", [])]
        connections = np.asarray(payload_copy.pop("connections", []), dtype=np.int32).reshape(-1, 2)
        metrics = np.asarray(payload_copy.pop("metrics", []), dtype=np.float64).reshape(-1)
        energy_traces = [
            np.asarray(trace, dtype=np.float64) for trace in payload_copy.pop("energy_traces", [])
        ]
        scale_traces = [
            np.asarray(trace, dtype=np.int16) for trace in payload_copy.pop("scale_traces", [])
        ]
        origin_indices = np.asarray(payload_copy.pop("origin_indices", []), dtype=np.int32).reshape(
            -1
        )
        connection_sources = [str(value) for value in payload_copy.pop("connection_sources", [])]
        frontier_lifecycle_events = list(payload_copy.pop("frontier_lifecycle_events", []) or [])
        diagnostics = dict(payload_copy.pop("diagnostics", {}) or {})
        return cls(
            traces=traces,
            connections=connections,
            metrics=metrics,
            energy_traces=energy_traces,
            scale_traces=scale_traces,
            origin_indices=origin_indices,
            connection_sources=connection_sources,
            frontier_lifecycle_events=frontier_lifecycle_events,
            diagnostics=diagnostics,
            extra=payload_copy,
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize to the legacy candidate dict shape used by checkpoints."""
        payload: dict[str, Any] = {
            "traces": self.traces,
            "connections": self.connections,
            "metrics": self.metrics,
            "energy_traces": self.energy_traces,
            "scale_traces": self.scale_traces,
            "origin_indices": self.origin_indices,
            "connection_sources": self.connection_sources,
            "diagnostics": self.diagnostics,
            **self.extra,
        }
        if self.frontier_lifecycle_events:
            payload["frontier_lifecycle_events"] = self.frontier_lifecycle_events
        return payload

    def apply_to_dict(self, target: dict[str, Any]) -> None:
        """Overwrite a legacy dict shell so it matches this manifest."""
        updated = self.to_payload()
        for key in list(target.keys()):
            if key not in updated:
                del target[key]
        target.update(updated)

    def append_unit(self, unit_payload: dict[str, Any]) -> None:
        """Append a per-origin candidate unit into this manifest."""
        unit_traces = [np.asarray(trace, dtype=np.float32) for trace in unit_payload["traces"]]
        unit_connections = np.asarray(unit_payload["connections"], dtype=np.int32).reshape(-1, 2)
        unit_metrics = np.asarray(unit_payload["metrics"], dtype=np.float32).reshape(-1)
        unit_origin_indices = np.asarray(
            unit_payload.get("origin_indices", []), dtype=np.int32
        ).reshape(-1)
        unit_connection_sources = normalize_candidate_connection_sources(
            unit_payload.get("connection_sources"),
            len(unit_connections),
            default_source=str(unit_payload.get("candidate_source", "unknown")),
        )
        base_candidate_index = len(self.traces)
        emitted_frontier_count = 0
        for raw_event in unit_payload.get("frontier_lifecycle_events", []):
            if not isinstance(raw_event, dict):
                continue
            event = dict(raw_event)
            if event.get("survived_candidate_manifest"):
                event["manifest_candidate_index"] = base_candidate_index + emitted_frontier_count
                emitted_frontier_count += 1
            else:
                event["manifest_candidate_index"] = None
            self.frontier_lifecycle_events.append(event)

        self.traces.extend(unit_traces)
        self.energy_traces.extend(
            np.asarray(trace, dtype=np.float32) for trace in unit_payload["energy_traces"]
        )
        self.scale_traces.extend(
            np.asarray(trace, dtype=np.int16) for trace in unit_payload["scale_traces"]
        )

        if unit_connections.size:
            self.connections = (
                unit_connections
                if self.connections.size == 0
                else np.vstack([self.connections, unit_connections])
            )
            self.metrics = np.concatenate([self.metrics, unit_metrics])
            self.origin_indices = np.concatenate([self.origin_indices, unit_origin_indices])
            self.connection_sources.extend(unit_connection_sources)

        _merge_edge_diagnostics(self.diagnostics, unit_payload.get("diagnostics", {}))

    @classmethod
    def append_unit_into_dict(
        cls,
        target: dict[str, Any],
        unit_payload: dict[str, Any],
    ) -> None:
        """Append a unit into a legacy dict shell (resumable unit aggregation)."""
        manifest = cls.from_payload(target)
        manifest.append_unit(unit_payload)
        manifest.apply_to_dict(target)

    def reordered(self, sort_order: np.ndarray) -> CandidateManifest:
        """Return a new manifest with candidates reordered by the provided indices."""
        sort_idx = np.asarray(sort_order, dtype=np.int32).reshape(-1)
        if sort_idx.size == 0:
            return self

        return CandidateManifest(
            traces=[self.traces[i] for i in sort_idx.tolist()],
            connections=np.asarray(self.connections[sort_idx], dtype=np.int32).reshape(-1, 2),
            metrics=np.asarray(self.metrics[sort_idx], dtype=np.float64),
            energy_traces=[self.energy_traces[i] for i in sort_idx.tolist()],
            scale_traces=[self.scale_traces[i] for i in sort_idx.tolist()],
            origin_indices=np.asarray(self.origin_indices[sort_idx], dtype=np.int32),
            connection_sources=[self.connection_sources[i] for i in sort_idx.tolist()],
            frontier_lifecycle_events=list(self.frontier_lifecycle_events),
            diagnostics=dict(self.diagnostics),
            extra=dict(self.extra),
        )

    def endpoint_pair_set(self) -> set[tuple[int, int]]:
        """Return orientation-independent terminal endpoint pairs."""
        pairs: set[tuple[int, int]] = set()
        normalized = np.asarray(self.connections, dtype=np.int32).reshape(-1, 2)
        for start_vertex, end_vertex in normalized:
            if int(start_vertex) < 0 or int(end_vertex) < 0:
                continue
            u, v = int(start_vertex), int(end_vertex)
            pairs.add((u, v) if u < v else (v, u))
        return pairs

    def frontier_origin_counts(self) -> dict[int, int]:
        """Count candidates tagged connection_source ``frontier`` per origin index."""
        counts: dict[int, int] = {}
        for index, origin_index in enumerate(self.origin_indices.reshape(-1)):
            if index >= len(self.connection_sources):
                continue
            if self.connection_sources[index] != "frontier":
                continue
            origin_key = int(origin_index)
            counts[origin_key] = counts.get(origin_key, 0) + 1
        return counts


# Legacy free-function names (dict-shell seams used by resumable / older call sites).
def _apply_manifest_to_dict(target: dict[str, Any], manifest: CandidateManifest) -> None:
    """Write manifest fields back into a legacy dict payload shell."""
    manifest.apply_to_dict(target)


def _append_candidate_unit(target: dict[str, Any], unit_payload: dict[str, Any]) -> None:
    """Append a per-origin candidate payload into the aggregate candidate dict shell."""
    CandidateManifest.append_unit_into_dict(target, unit_payload)


__all__ = [
    "CandidateManifest",
    "_append_candidate_unit",
    "_apply_manifest_to_dict",
    "candidate_as_payload",
    "endpoint_pairs_from_connections",
    "incident_pair_counts",
    "normalize_candidate_connection_sources",
    "reorder_candidate_payload",
]
