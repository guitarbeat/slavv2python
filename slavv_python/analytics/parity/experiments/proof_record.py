"""Load proof JSON only when dest_run_root matches the folder on disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slavv_python.analytics.parity.constants import ANALYSIS_DIR


class ProofRecordError(ValueError):
    """Raised when a proof JSON cannot be cited for the opened folder."""


@dataclass(frozen=True)
class ProofRecord:
    """A proof JSON that is paired to the run folder it sits under."""

    path: Path
    run_root: Path
    dest_run_root: Path
    source_run_root: Path | None
    passed: bool | None
    stages: tuple[str, ...]
    adr0012_evaluated: bool | None
    payload: dict[str, Any]


def run_root_from_proof_path(path: Path) -> Path:
    """Return the Parity Run root that owns a ``03_Analysis`` proof file."""
    resolved = path.expanduser().resolve()
    analysis_name = ANALYSIS_DIR.name
    for parent in (resolved.parent, *resolved.parents):
        if parent.name == analysis_name:
            return parent.parent
    raise ProofRecordError(f"proof JSON is not under {analysis_name}/: {resolved}")


def load_proof_record(
    path: Path,
    *,
    expected_run_root: Path | None = None,
) -> ProofRecord:
    """Load a proof JSON and refuse dest/folder mismatch.

    A file can sit under ``crop_M_exact_v3/03_Analysis`` and still belong to
    ``crop_M_exact``. Citation must go through this seam.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ProofRecordError(f"proof JSON not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProofRecordError(f"proof JSON is not an object: {resolved}")

    run_root = run_root_from_proof_path(resolved)
    dest_raw = payload.get("dest_run_root")
    if not isinstance(dest_raw, str) or not dest_raw:
        raise ProofRecordError(f"proof JSON missing dest_run_root: {resolved}")
    dest_run_root = Path(dest_raw).expanduser().resolve()
    if dest_run_root != run_root:
        raise ProofRecordError(f"dest_run_root {dest_run_root} does not match folder {run_root}")
    if expected_run_root is not None and expected_run_root.expanduser().resolve() != run_root:
        raise ProofRecordError(f"expected run root {expected_run_root} does not match {run_root}")

    source_raw = payload.get("source_run_root")
    source_run_root = (
        Path(source_raw).expanduser().resolve()
        if isinstance(source_raw, str) and source_raw
        else None
    )
    gate = payload.get("edges_adr0012_gate")
    evaluated: bool | None = None
    if isinstance(gate, dict) and "adr0012_evaluated" in gate:
        evaluated = bool(gate.get("adr0012_evaluated"))

    stages_raw = payload.get("stages", ())
    stages: tuple[str, ...]
    if isinstance(stages_raw, str):
        stages = (stages_raw,)
    elif isinstance(stages_raw, list):
        stages = tuple(str(item) for item in stages_raw)
    else:
        stages = ()

    passed = payload.get("passed")
    return ProofRecord(
        path=resolved,
        run_root=run_root,
        dest_run_root=dest_run_root,
        source_run_root=source_run_root,
        passed=None if passed is None else bool(passed),
        stages=stages,
        adr0012_evaluated=evaluated,
        payload=payload,
    )


def require_evaluated_adr0012(record: ProofRecord, *, stage: str) -> None:
    """Refuse Edges/Network citations that did not evaluate ADR 0012."""
    if stage not in {"edges", "network"}:
        return
    if record.adr0012_evaluated is not True:
        raise ProofRecordError(
            f"{stage} proof is not an evaluated ADR 0012 citation "
            f"(adr0012_evaluated={record.adr0012_evaluated})"
        )
