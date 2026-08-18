"""Read-only Phase 2 profiling baseline from a frozen Phase 1 run dest.

Does not launch writers, unwind Fortran order, or change production defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slavv_python.analytics.parity.constants import LIVE_DEST_NAMES

PIPELINE_STAGES = ("energy", "vertices", "edges", "network")
CARRIED_REASON = "elapsed_seconds=0 means cache-resumed / carried lineage, not instant compute"
ENERGY_HISTORICAL_NOTE = (
    "Energy wall-clock is not measured on the frozen dest. Historical n_jobs=6 "
    "throughput lives in docs/solutions/parity/exact-energy-chunk-parallelism.md "
    "and is not this dest's stage_metrics."
)


@dataclass(frozen=True)
class StageTiming:
    """One pipeline stage from ``run_manifest.stage_metrics``."""

    name: str
    elapsed_seconds: float
    peak_memory_bytes: int | None
    status: str
    measured: bool
    completed_at: str | None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "status": self.status,
            "measured": self.measured,
            "completed_at": self.completed_at,
        }
        if not self.measured:
            payload["reason"] = CARRIED_REASON
        return payload


def is_measured_elapsed(elapsed_seconds: float) -> bool:
    """True when the dest recorded a positive stage wall-clock."""
    return float(elapsed_seconds) > 0.0


def peak_memory_mib(peak_memory_bytes: int | None) -> float | None:
    """Convert peak RSS bytes to MiB."""
    if peak_memory_bytes is None:
        return None
    return float(peak_memory_bytes) / (1024.0 * 1024.0)


def parse_stage_metrics(stage_metrics: dict[str, Any]) -> list[StageTiming]:
    """Parse Energy→Network timings; ignore preprocess calendar spans."""
    records: list[StageTiming] = []
    for name in PIPELINE_STAGES:
        raw = stage_metrics.get(name)
        if not isinstance(raw, dict):
            records.append(
                StageTiming(
                    name=name,
                    elapsed_seconds=0.0,
                    peak_memory_bytes=None,
                    status="missing",
                    measured=False,
                    completed_at=None,
                )
            )
            continue
        elapsed = float(raw.get("elapsed_seconds") or 0.0)
        peak = raw.get("peak_memory_bytes")
        peak_i = int(peak) if peak is not None else None
        records.append(
            StageTiming(
                name=name,
                elapsed_seconds=elapsed,
                peak_memory_bytes=peak_i,
                status=str(raw.get("status") or "unknown"),
                measured=is_measured_elapsed(elapsed),
                completed_at=str(raw["completed_at"]) if raw.get("completed_at") else None,
            )
        )
    return records


def bottleneck_measured(records: list[StageTiming]) -> str | None:
    """Slowest stage that actually recorded wall-clock on this dest."""
    measured = [item for item in records if item.measured]
    if not measured:
        return None
    return max(measured, key=lambda item: item.elapsed_seconds).name


def baseline_payload(
    *,
    records: list[StageTiming],
    n_jobs: int | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tracked/scratch JSON. Not a stretch or unwind claim."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "phase": 2,
        "workstream": "profiling_baseline",
        "isolation_only": False,
        "not_unwind": True,
        "not_stretch": True,
        "do_not_overwrite": list(LIVE_DEST_NAMES),
        "n_jobs": n_jobs,
        "stages": {item.name: item.to_dict() for item in records},
        "bottleneck_measured_on_dest": bottleneck_measured(records),
        "bottleneck_full_pipeline_historical": "energy",
        "energy_historical_note": ENERGY_HISTORICAL_NOTE,
        "next_allowed": (
            "Energy --n-jobs auto is implemented (opt-in; dest default stays 1). "
            "Next: Edges/Network profiling on an authorized writer. "
            "Fortran-order unwind still needs an explicit Phase 2 ADR."
        ),
    }
    if extra:
        payload.update(extra)
    return payload


__all__ = [
    "CARRIED_REASON",
    "ENERGY_HISTORICAL_NOTE",
    "PIPELINE_STAGES",
    "StageTiming",
    "baseline_payload",
    "bottleneck_measured",
    "is_measured_elapsed",
    "parse_stage_metrics",
    "peak_memory_mib",
]
