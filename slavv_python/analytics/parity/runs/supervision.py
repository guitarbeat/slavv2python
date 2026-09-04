"""Stateful supervision policy for parity-run writers.

This module owns lifecycle decisions; the JSON manifest, writer lease, and job
registry remain persistence/process adapters.  Keeping the policy here gives
pause, resume, and monitor callers one transition vocabulary without changing
the artifacts they already read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from slavv_python.analytics.parity.runs import parity_job_lifecycle as lifecycle


class InvalidParityJobTransitionError(RuntimeError):
    """Raised when a run-local job manifest receives an unsafe transition."""


_ALLOWED: dict[str | None, frozenset[str]] = {
    None: frozenset({"launched", "running", "interrupted"}),
    "launched": frozenset({"running", "failed", "interrupted"}),
    "running": frozenset({"succeeded", "failed", "interrupted"}),
    "succeeded": frozenset({"succeeded"}),
    "failed": frozenset({"failed"}),
    "interrupted": frozenset({"interrupted", "running"}),
}


@dataclass(frozen=True)
class ParityRunObservation:
    """Read-only view of the lifecycle fields used by supervision."""

    run_dir: Path
    status: str | None
    stage: str | None
    pid: int | None
    checkpoint: str | None = None


class ParityRunSupervisor:
    """Coordinate safe lifecycle transitions for one parity run root."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.expanduser().resolve()

    def observe(self) -> ParityRunObservation:
        manifest = lifecycle.load_parity_job_manifest(self.run_dir) or {}
        pid = manifest.get("pid")
        return ParityRunObservation(
            run_dir=self.run_dir,
            status=_string_or_none(manifest.get("status")),
            stage=_string_or_none(manifest.get("stage")),
            pid=int(pid) if isinstance(pid, int | str) and str(pid).isdigit() else None,
            checkpoint=_checkpoint_from(manifest),
        )

    def transition(self, status: str, **updates: Any) -> dict[str, Any]:
        """Persist *status* only when the current state permits it."""
        if status not in lifecycle.TERMINAL_STATUSES | lifecycle.ACTIVE_STATUSES:
            raise ValueError(f"unsupported parity job status: {status}")
        current = self.observe().status
        if status not in _ALLOWED.get(current, frozenset()):
            raise InvalidParityJobTransitionError(
                f"cannot transition parity job {current!r} -> {status!r} "
                f"for {self.run_dir}"
            )
        return lifecycle.update_parity_job_manifest(self.run_dir, status=status, **updates)

    def start(self, *, pid: int, command: str | list[str], stage: str) -> dict[str, Any]:
        """Enter the running state, preserving the existing start timestamp."""
        current = self.observe().status
        if current == "running":
            existing = lifecycle.load_parity_job_manifest(self.run_dir)
            return existing or lifecycle.mark_parity_job_running(
                self.run_dir, pid=pid, command=command, stage=stage
            )
        if current not in (None, "interrupted", "launched"):
            raise InvalidParityJobTransitionError(
                f"cannot transition parity job {current!r} -> 'running' for {self.run_dir}"
            )
        return lifecycle.mark_parity_job_running(
            self.run_dir, pid=pid, command=command, stage=stage
        )

    def finish(self, *, status: str, exit_code: int | None = None, reason: str = "") -> dict[str, Any]:
        """Enter an idempotent terminal state with durable completion metadata."""
        if status not in lifecycle.TERMINAL_STATUSES:
            raise ValueError(f"unsupported terminal parity job status: {status}")
        updates: dict[str, Any] = {"ended_at": lifecycle.now_iso(), "exit_code": exit_code}
        if reason:
            updates["reason"] = reason
        return self.transition(status, **updates)


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _checkpoint_from(manifest: dict[str, Any]) -> str | None:
    for key in ("last_checkpoint", "checkpoint", "current_stage", "stage"):
        value = _string_or_none(manifest.get(key))
        if value:
            return value
    return None


if TYPE_CHECKING:
    from pathlib import Path

InvalidParityJobTransition = InvalidParityJobTransitionError

__all__ = ["InvalidParityJobTransition", "InvalidParityJobTransitionError", "ParityRunObservation", "ParityRunSupervisor"]
