"""Persistence and observability seam for an Edges execution."""

from __future__ import annotations

from typing import Any

from slavv_python.engine.state.io import atomic_joblib_dump, atomic_write_json
from slavv_python.pipeline.edges.candidate_manifest import CandidateManifest, candidate_as_payload


class EdgeExecutionArtifacts:
    """Write durable Edges artifacts without exposing storage to the coordinator.

    The artifact writer is deliberately small: ``EdgeManager`` remains the only
    public Edges interface, while checkpoint names and payloads stay unchanged.
    """

    def __init__(self, artifact_path: Any, *, authorized: bool) -> None:
        self._artifact_path = artifact_path
        self.authorized = authorized

    def write_candidates(self, manifest: CandidateManifest, audit: dict[str, Any]) -> None:
        """Persist the Candidate Set and its audit when the writer is authorized."""
        if not self.authorized:
            return
        atomic_write_json(self._artifact_path("candidate_audit.json"), audit)
        atomic_joblib_dump(
            candidate_as_payload(manifest),
            self._artifact_path("candidates.pkl"),
        )

    def write_timing(self, payload: dict[str, Any]) -> None:
        """Persist the durable split timing artifact."""
        if self.authorized:
            from slavv_python.analytics.performance.edge_timing import write_edge_timing

            write_edge_timing(self._artifact_path("phase2_edges_split.json"), payload)

    def write_final(
        self,
        edge_payload: dict[str, Any],
        *,
        candidate_lifecycle: dict[str, Any] | None = None,
    ) -> None:
        """Persist finalized Edges artifacts and optional lifecycle diagnostics."""
        if not self.authorized:
            return
        if candidate_lifecycle is not None:
            atomic_write_json(
                self._artifact_path("candidate_lifecycle.json"),
                candidate_lifecycle,
            )
        atomic_joblib_dump(edge_payload, self._artifact_path("chosen_edges.pkl"))


__all__ = ["EdgeExecutionArtifacts"]
