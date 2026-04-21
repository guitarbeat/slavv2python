"""Helpers for resume-guard fingerprint checks and mismatch handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .constants import STATUS_BLOCKED, STATUS_PENDING
from .models import _now_iso

if TYPE_CHECKING:
    import logging

    from .models import RunSnapshot


def fingerprint_mismatches(
    snapshot: RunSnapshot,
    *,
    input_fingerprint: str,
    params_fingerprint: str,
) -> list[str]:
    """Return the fingerprint dimensions that no longer match the saved run."""
    mismatch: list[str] = []
    if snapshot.input_fingerprint and snapshot.input_fingerprint != input_fingerprint:
        mismatch.append("input")
    if snapshot.params_fingerprint and snapshot.params_fingerprint != params_fingerprint:
        mismatch.append("parameters")
    return mismatch


def update_snapshot_fingerprints(
    snapshot: RunSnapshot,
    *,
    input_fingerprint: str,
    params_fingerprint: str,
) -> None:
    """Persist the current input and parameter fingerprints onto the snapshot."""
    snapshot.input_fingerprint = input_fingerprint
    snapshot.params_fingerprint = params_fingerprint


def apply_resume_reset(
    snapshot: RunSnapshot,
    *,
    input_fingerprint: str,
    params_fingerprint: str,
    mismatch: list[str],
    logger: logging.Logger,
) -> None:
    """Mark a snapshot as reset after an allowed rerun from the energy boundary."""
    logger.info(
        "Resuming with explicit rerun from energy after %s change",
        ", ".join(mismatch),
    )
    update_snapshot_fingerprints(
        snapshot,
        input_fingerprint=input_fingerprint,
        params_fingerprint=params_fingerprint,
    )
    snapshot.status = STATUS_PENDING
    snapshot.last_event = "Pipeline reset after resume guard mismatch"


def build_resume_block_message(mismatch: list[str]) -> str:
    """Build the standard error message for a blocked resume mismatch."""
    return (
        "Resume blocked because the "
        + " and ".join(mismatch)
        + " fingerprint changed. Re-run with force_rerun_from='energy' to start a fresh pipeline."
    )


def apply_resume_block(snapshot: RunSnapshot, mismatch: list[str]) -> str:
    """Mark the snapshot as blocked and append a resume-guard error entry."""
    message = build_resume_block_message(mismatch)
    snapshot.status = STATUS_BLOCKED
    snapshot.last_event = message
    snapshot.errors.append({"message": message, "at": _now_iso()})
    return message


__all__ = [
    "apply_resume_block",
    "apply_resume_reset",
    "build_resume_block_message",
    "fingerprint_mismatches",
    "update_snapshot_fingerprints",
]
