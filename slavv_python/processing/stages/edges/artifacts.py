"""Optional persistence hooks for edge-stage artifacts (parity checkpoints, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class EdgeCandidatePersistence(Protocol):
    """Writes parity-oriented edge candidate checkpoints after discovery."""

    def write_candidate_checkpoint(
        self,
        checkpoints_dir: Path,
        candidates: dict[str, Any],
    ) -> None:
        """Persist a slim candidate snapshot for exact-route replay and fail-fast."""


class NoOpEdgeCandidatePersistence:
    """Default hook: no cross-run parity checkpoint."""

    def write_candidate_checkpoint(
        self,
        checkpoints_dir: Path,
        candidates: dict[str, Any],
    ) -> None:
        del checkpoints_dir, candidates


NO_OP_EDGE_CANDIDATE_PERSISTENCE = NoOpEdgeCandidatePersistence()


def resolve_edge_candidate_persistence(
    params: dict[str, Any],
    *,
    use_frontier: bool,
) -> EdgeCandidatePersistence:
    """Return parity checkpoint writer only for exact-route frontier resumable runs."""
    if not use_frontier or not bool(params.get("comparison_exact_network", False)):
        return NO_OP_EDGE_CANDIDATE_PERSISTENCE
    from slavv_python.analytics.parity.edge_artifacts import (
        ParityEdgeCandidatePersistence,
    )

    return ParityEdgeCandidatePersistence()


__all__ = [
    "EdgeCandidatePersistence",
    "NO_OP_EDGE_CANDIDATE_PERSISTENCE",
    "NoOpEdgeCandidatePersistence",
    "resolve_edge_candidate_persistence",
]
