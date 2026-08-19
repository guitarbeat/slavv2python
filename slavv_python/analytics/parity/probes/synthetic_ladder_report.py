"""Synthetic complexity ladder report assembly and soft-cap stop policy.

Pure helpers used by ``scripts/ladder/run.py`` so unit tests
can cover AE1-AE3 without MATLAB.
"""

from __future__ import annotations

from typing import Any, Literal

from slavv_python.utils.synthetic import LADDER_RUNG_IDS, LADDER_RUNG_MAX_DIM

NON_CERTIFICATION_NOTE = (
    "Synthetic complexity ladder - NOT Certification / NOT Phase 1. "
    "Do not update ONE TRUTH or claim-run roots from this report."
)

LadderOutcome = Literal["first_break", "soft_cap_full_match", "inconclusive", "failed"]
SoftCapReason = Literal["size", "time", "end_of_ladder"]

DEFAULT_SOFT_TIME_SEC = 180.0
DEFAULT_SOFT_SIZE_MAX_DIM = 64


def soft_cap_blocks_next_rung(
    *,
    next_rung_id: str | None,
    prior_matlab_wall_sec: float | None,
    prior_python_wall_sec: float | None,
    soft_time_sec: float = DEFAULT_SOFT_TIME_SEC,
    soft_size_max_dim: int = DEFAULT_SOFT_SIZE_MAX_DIM,
) -> SoftCapReason | None:
    """Return soft-cap reason if the next rung must not start; else None.

    Soft-time is enforced pre-start of the next rung (A-plan1). When a side's
    wall_sec is null (reuse), that side is skipped for the time budget check.
    """
    if next_rung_id is None:
        return "end_of_ladder"
    max_dim = LADDER_RUNG_MAX_DIM.get(next_rung_id)
    if max_dim is None:
        return "end_of_ladder"
    if max_dim > soft_size_max_dim:
        return "size"
    for wall in (prior_matlab_wall_sec, prior_python_wall_sec):
        if wall is not None and float(wall) > soft_time_sec:
            return "time"
    return None


def assemble_ladder_report(
    *,
    rung_results: list[dict[str, Any]],
    outcome: LadderOutcome,
    first_break_rung: str | None = None,
    first_break_surface: str | None = None,
    soft_cap_reason: SoftCapReason | None = None,
    created_utc: str,
    soft_time_sec: float = DEFAULT_SOFT_TIME_SEC,
    soft_size_max_dim: int = DEFAULT_SOFT_SIZE_MAX_DIM,
) -> dict[str, Any]:
    """Build the durable ladder_report.json payload."""
    return {
        "created_utc": created_utc,
        "note": NON_CERTIFICATION_NOTE,
        "outcome": outcome,
        "first_break_rung": first_break_rung,
        "first_break_surface": first_break_surface,
        "soft_cap_reason": soft_cap_reason,
        "soft_policy": {
            "soft_time_sec_per_side": soft_time_sec,
            "soft_size_max_dim": soft_size_max_dim,
            "fixed_rung_ids": list(LADDER_RUNG_IDS),
        },
        "ladder_rungs": list(rung_results),
    }


def orchestrate_from_rung_results(
    rung_results: list[dict[str, Any]],
    *,
    created_utc: str,
    soft_time_sec: float = DEFAULT_SOFT_TIME_SEC,
    soft_size_max_dim: int = DEFAULT_SOFT_SIZE_MAX_DIM,
    planned_rung_ids: tuple[str, ...] | list[str] = LADDER_RUNG_IDS,
) -> dict[str, Any]:
    """Derive ladder outcome from an ordered list of per-rung result dicts.

    Each rung result must include:
      - rung_id
      - status: ``match`` | ``first_break`` | ``inconclusive`` | ``failed``
      - first_break_surface (when status is first_break)
      - matlab_wall_sec / python_wall_sec (optional; null under reuse)
      - executed: bool

    Callers that stop early should only include executed rungs. Soft-cap after a
    full match series is inferred from whether planned rungs remain and why they
    were not started (``soft_cap_blocked`` on the last executed rung, or
    end_of_ladder when all planned rungs matched).
    """
    if not rung_results:
        return assemble_ladder_report(
            rung_results=[],
            outcome="inconclusive",
            created_utc=created_utc,
            soft_time_sec=soft_time_sec,
            soft_size_max_dim=soft_size_max_dim,
        )

    for rung in rung_results:
        status = rung.get("status")
        if status == "first_break":
            return assemble_ladder_report(
                rung_results=rung_results,
                outcome="first_break",
                first_break_rung=str(rung["rung_id"]),
                first_break_surface=rung.get("first_break_surface"),
                created_utc=created_utc,
                soft_time_sec=soft_time_sec,
                soft_size_max_dim=soft_size_max_dim,
            )
        if status in {"inconclusive", "failed"}:
            return assemble_ladder_report(
                rung_results=rung_results,
                outcome=status,
                created_utc=created_utc,
                soft_time_sec=soft_time_sec,
                soft_size_max_dim=soft_size_max_dim,
            )

    # All executed rungs matched.
    last = rung_results[-1]
    blocked = last.get("soft_cap_blocked")
    if blocked in {"size", "time", "end_of_ladder"}:
        return assemble_ladder_report(
            rung_results=rung_results,
            outcome="soft_cap_full_match",
            soft_cap_reason=blocked,
            created_utc=created_utc,
            soft_time_sec=soft_time_sec,
            soft_size_max_dim=soft_size_max_dim,
        )

    executed_ids = [r["rung_id"] for r in rung_results]
    remaining = [rid for rid in planned_rung_ids if rid not in executed_ids]
    if not remaining:
        return assemble_ladder_report(
            rung_results=rung_results,
            outcome="soft_cap_full_match",
            soft_cap_reason="end_of_ladder",
            created_utc=created_utc,
            soft_time_sec=soft_time_sec,
            soft_size_max_dim=soft_size_max_dim,
        )

    reason = soft_cap_blocks_next_rung(
        next_rung_id=remaining[0],
        prior_matlab_wall_sec=last.get("matlab_wall_sec"),
        prior_python_wall_sec=last.get("python_wall_sec"),
        soft_time_sec=soft_time_sec,
        soft_size_max_dim=soft_size_max_dim,
    )
    return assemble_ladder_report(
        rung_results=rung_results,
        outcome="soft_cap_full_match" if reason else "inconclusive",
        soft_cap_reason=reason,
        created_utc=created_utc,
        soft_time_sec=soft_time_sec,
        soft_size_max_dim=soft_size_max_dim,
    )
