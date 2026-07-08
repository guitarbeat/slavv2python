"""Optional persistence hooks for edge-stage artifacts (parity checkpoints, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class EdgeCandidatePersistence(Protocol):
    """Writes parity-oriented edge candidate checkpoints after discovery."""

    def write_candidate_checkpoint(
        self,
        checkpoints_dir: Path,
        candidates: dict[str, Any],
        *,
        include_debug_maps: bool = False,
    ) -> dict[str, Any] | None:
        """Persist a slim candidate snapshot for exact-route replay and fail-fast."""


class NoOpEdgeCandidatePersistence:
    """Default hook: no cross-run parity checkpoint."""

    def write_candidate_checkpoint(
        self,
        checkpoints_dir: Path,
        candidates: dict[str, Any],
        *,
        include_debug_maps: bool = False,
    ) -> dict[str, Any] | None:
        del checkpoints_dir, candidates, include_debug_maps
        return None


NO_OP_EDGE_CANDIDATE_PERSISTENCE = NoOpEdgeCandidatePersistence()


def resolve_edge_candidate_persistence(
    params: dict[str, Any],
    *,
    use_frontier: bool,
) -> EdgeCandidatePersistence:
    """Return parity checkpoint writer only for exact-route frontier resumable runs."""
    if not use_frontier or not bool(params.get("comparison_exact_network", False)):
        return NO_OP_EDGE_CANDIDATE_PERSISTENCE
    from slavv_python.analytics.parity.probes.edge_artifacts import (
        ParityEdgeCandidatePersistence,
    )

    return ParityEdgeCandidatePersistence()


__all__ = [
    "NO_OP_EDGE_CANDIDATE_PERSISTENCE",
    "EdgeCandidatePersistence",
    "NoOpEdgeCandidatePersistence",
    "resolve_edge_candidate_persistence",
]
