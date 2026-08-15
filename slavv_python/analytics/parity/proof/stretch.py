"""True zero-tolerance stretch status and crop→full unlock helpers.

Phase 1 Certification (ONE TRUTH CLOSED) is unchanged by this module.
Stretch proofs use ``--strict-floats`` and a hard unlock token; never rewrite
ADR 0011/0012 ship bars or ONE TRUTH CLOSED language.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class StretchStatus(str, Enum):
    """Authoritative stretch progress / failure taxonomy (KTD5)."""

    STRETCH_NOT_STARTED = "stretch_not_started"
    FLOAT_PATH_BUILDING = "float_path_building"
    INCOMPLETE_INFRA = "incomplete_infra"
    CROP_ENERGY_RUNNING = "crop_energy_running"
    BLOCKED_FLOAT_PATH = "blocked_float_path"
    CROP_ENERGY_PASSED = "crop_energy_passed"
    INCOMPLETE_DISCRETE = "incomplete_discrete"
    FULL_REFUSED = "full_refused"
    FULL_RUNNING = "full_running"
    INCOMPLETE_AT_FULL = "incomplete_at_full"
    STRETCH_COMPLETE = "stretch_complete"


class StretchFieldSet(str, Enum):
    """Field set named by a crop unlock token."""

    ENERGY = "energy"
    ENERGY_AND_DISCRETE = "energy+discrete"


class StretchFailureClass(str, Enum):
    """Operator-facing failure class before status mapping."""

    INFRA = "infra"
    FLOAT_PATH = "float_path"
    DISCRETE = "discrete"
    AT_FULL = "at_full"


UNLOCK_SCHEMA_VERSION = 1
STATUS_SCHEMA_VERSION = 1
UNLOCK_FILENAME = "stretch_crop_unlock.json"
STATUS_FILENAME = "stretch_status.json"

# Claim root must never be overwritten by stretch writers (KTD6).
PHASE1_CLAIM_RUN_NAME = "canonical_full_v18"


@dataclass(frozen=True)
class StretchUnlockToken:
    """Crop unlock authorizing full stretch for a matching field set."""

    schema_version: int
    field_set: StretchFieldSet
    crop_dest_run_root: str
    oracle_root: str
    proof_path: str
    strict_floats: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "field_set": self.field_set.value,
            "crop_dest_run_root": self.crop_dest_run_root,
            "oracle_root": self.oracle_root,
            "proof_path": self.proof_path,
            "strict_floats": self.strict_floats,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StretchUnlockToken:
        return cls(
            schema_version=int(payload["schema_version"]),
            field_set=StretchFieldSet(str(payload["field_set"])),
            crop_dest_run_root=str(payload["crop_dest_run_root"]),
            oracle_root=str(payload["oracle_root"]),
            proof_path=str(payload["proof_path"]),
            strict_floats=bool(payload.get("strict_floats", True)),
        )


@dataclass(frozen=True)
class StretchGateDecision:
    """Result of gating full-volume stretch entry."""

    allowed: bool
    status: StretchStatus
    reason: str


@dataclass(frozen=True)
class ClassifiedStretchFailure:
    """Mapped failure class → stretch status (never conflates infra with R3)."""

    status: StretchStatus
    failure_class: StretchFailureClass
    detail: str


def classify_stretch_failure(
    failure_class: StretchFailureClass,
    *,
    detail: str,
) -> ClassifiedStretchFailure:
    """Map a failure class to status; infra never becomes ``blocked_float_path``."""
    mapping = {
        StretchFailureClass.INFRA: StretchStatus.INCOMPLETE_INFRA,
        StretchFailureClass.FLOAT_PATH: StretchStatus.BLOCKED_FLOAT_PATH,
        StretchFailureClass.DISCRETE: StretchStatus.INCOMPLETE_DISCRETE,
        StretchFailureClass.AT_FULL: StretchStatus.INCOMPLETE_AT_FULL,
    }
    return ClassifiedStretchFailure(
        status=mapping[failure_class],
        failure_class=failure_class,
        detail=detail,
    )


def field_set_covers(have: StretchFieldSet, need: StretchFieldSet) -> bool:
    """Return True when ``have`` authorizes the requested ``need`` field set."""
    if need == StretchFieldSet.ENERGY:
        return have in {
            StretchFieldSet.ENERGY,
            StretchFieldSet.ENERGY_AND_DISCRETE,
        }
    return have == StretchFieldSet.ENERGY_AND_DISCRETE


def write_stretch_unlock(
    path: Path,
    *,
    field_set: StretchFieldSet,
    dest_run_root: Path,
    oracle_root: Path,
    proof_path: Path,
    strict_floats: bool = True,
) -> StretchUnlockToken:
    """Persist a crop unlock artifact (Energy or Energy+discrete)."""
    if not strict_floats:
        raise ValueError("stretch unlock requires strict_floats=True")
    token = StretchUnlockToken(
        schema_version=UNLOCK_SCHEMA_VERSION,
        field_set=field_set,
        crop_dest_run_root=str(Path(dest_run_root).resolve()),
        oracle_root=str(Path(oracle_root).resolve()),
        proof_path=str(Path(proof_path).resolve()),
        strict_floats=True,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(token.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return token


def load_stretch_unlock(path: Path) -> StretchUnlockToken:
    """Load a crop unlock token from disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"stretch unlock must be a JSON object: {path}")
    return StretchUnlockToken.from_dict(payload)


def gate_full_stretch_entry(
    *,
    unlock_path: Path,
    requested_field_set: StretchFieldSet,
    dest_run_root: Path,
    oracle_root: Path,
) -> StretchGateDecision:
    """Refuse full stretch unless unlock exists for the same field set / oracle."""
    dest = Path(dest_run_root)
    if dest.name == PHASE1_CLAIM_RUN_NAME or PHASE1_CLAIM_RUN_NAME in dest.parts:
        return StretchGateDecision(
            allowed=False,
            status=StretchStatus.FULL_REFUSED,
            reason=(
                f"refusing stretch dest that overwrites Phase 1 claim root "
                f"{PHASE1_CLAIM_RUN_NAME}"
            ),
        )

    path = Path(unlock_path)
    if not path.is_file():
        return StretchGateDecision(
            allowed=False,
            status=StretchStatus.FULL_REFUSED,
            reason="crop unlock artifact missing; run crop --strict-floats first",
        )

    try:
        token = load_stretch_unlock(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return StretchGateDecision(
            allowed=False,
            status=StretchStatus.FULL_REFUSED,
            reason=f"invalid crop unlock artifact: {exc}",
        )

    if not token.strict_floats:
        return StretchGateDecision(
            allowed=False,
            status=StretchStatus.FULL_REFUSED,
            reason="unlock was not produced under strict_floats",
        )

    if Path(token.oracle_root).resolve() != Path(oracle_root).resolve():
        return StretchGateDecision(
            allowed=False,
            status=StretchStatus.FULL_REFUSED,
            reason="unlock oracle_root does not match requested full oracle",
        )

    if not field_set_covers(token.field_set, requested_field_set):
        need = requested_field_set.value
        have = token.field_set.value
        return StretchGateDecision(
            allowed=False,
            status=StretchStatus.FULL_REFUSED,
            reason=(
                f"unlock field_set={have!r} does not authorize requested "
                f"field_set={need!r} (energy-only unlock cannot claim discrete)"
            ),
        )

    return StretchGateDecision(
        allowed=True,
        status=StretchStatus.FULL_RUNNING,
        reason="crop unlock matches requested field set and oracle",
    )


def write_stretch_status(
    path: Path,
    *,
    status: StretchStatus,
    note: str = "",
    findings_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write stretch status JSON without mutating ONE TRUTH CLOSED docs.

    ``findings_path`` is accepted only so callers can pass the findings file for
    AE4 checks; this function never writes to that path.
    """
    del findings_path  # intentionally unused — never mutate ONE TRUTH
    payload: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": status.value,
        "note": note,
        "phase1_claim_untouched": True,
        "phase1_claim_run": PHASE1_CLAIM_RUN_NAME,
    }
    if extra:
        payload["extra"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def mkl_spike_cannot_complete_stretch(*, mkl_bit_equal: bool) -> bool:
    """Policy: an MKL falsifier pass alone never means Approach A / stretch complete."""
    del mkl_bit_equal
    return True
