"""Parity-run edge artifact writers (kept out of production EdgeManager imports)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slavv_python.analytics.parity.constants import EDGE_CANDIDATE_CHECKPOINT_PATH
from slavv_python.analytics.parity.probes.matlab_fail_fast import build_candidate_snapshot_payload
from slavv_python.engine.state.tracker import atomic_joblib_dump

if TYPE_CHECKING:
    from pathlib import Path


class ParityEdgeCandidatePersistence:
    """Writes normalized edge candidate snapshots for prove-exact and fail-fast."""

    def write_candidate_checkpoint(
        self,
        checkpoints_dir: Path,
        candidates: dict[str, Any],
        *,
        include_debug_maps: bool = False,
    ) -> dict[str, Any]:
        snapshot_payload = build_candidate_snapshot_payload(
            candidates,
            include_debug_maps=include_debug_maps,
        )
        atomic_joblib_dump(
            snapshot_payload,
            checkpoints_dir / EDGE_CANDIDATE_CHECKPOINT_PATH.name,
        )
        return snapshot_payload


__all__ = ["ParityEdgeCandidatePersistence"]
