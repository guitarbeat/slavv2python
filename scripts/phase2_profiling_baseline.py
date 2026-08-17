#!/usr/bin/env python
"""Record a read-only Phase 2 profiling baseline from the frozen claim dest.

Does not launch writers, overwrite protected dests, or unwind Fortran order.
Writes scratch JSON. Optionally copies the same payload to a tracked docs path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slavv_python.analytics.parity.constants import (
    CHECKPOINTS_DIR,
    PHASE1_CLAIM_RUN_NAME,
    RUN_MANIFEST_PATH,
    VALIDATED_PARAMS_PATH,
)
from slavv_python.analytics.performance.phase2_baseline import (
    baseline_payload,
    parse_stage_metrics,
)
from slavv_python.pipeline.energy.matlab_engine_backend import (
    refuse_protected_stretch_energy_dest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "workspace" / "runs" / "oracle_180709_E" / PHASE1_CLAIM_RUN_NAME
DEFAULT_SCRATCH = REPO_ROOT / "workspace" / "scratch" / "phase2_profiling_baseline.json"
DEFAULT_TRACKED = REPO_ROOT / "docs" / "reference" / "core" / "phase2-profiling-baseline.json"
FREEZE_REL = "docs/reference/core/phase1-baseline-freeze.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _incomplete(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": 2,
        "workstream": "profiling_baseline",
        "result": "skip",
        "reason": reason,
        "not_unwind": True,
        "not_stretch": True,
    }


def _checkpoint_sizes(dest: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for stage in ("energy", "vertices", "edges", "network"):
        path = dest / CHECKPOINTS_DIR / f"checkpoint_{stage}.pkl"
        if path.is_file():
            sizes[stage] = int(path.stat().st_size)
    return sizes


def run_phase2_profiling_baseline(*, dest: Path) -> dict[str, Any]:
    manifest_path = dest / RUN_MANIFEST_PATH
    params_path = dest / VALIDATED_PARAMS_PATH
    try:
        manifest = _load_json(manifest_path)
        params = _load_json(params_path) if params_path.is_file() else {}
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        return _incomplete(str(exc))
    stage_metrics = manifest.get("stage_metrics")
    if not isinstance(stage_metrics, dict):
        return _incomplete("run_manifest.json missing stage_metrics")
    records = parse_stage_metrics(stage_metrics)
    n_jobs = params.get("n_jobs")
    n_jobs_i = int(n_jobs) if n_jobs is not None else None
    extra: dict[str, Any] = {
        "result": "ok",
        "claim_run_root": "workspace/runs/oracle_180709_E/canonical_full_v18",
        "freeze_ref": FREEZE_REL,
        "recorded_at": "2026-08-17",
        "checkpoint_bytes": _checkpoint_sizes(dest),
        "manifest_status": manifest.get("status"),
        "python_commit": manifest.get("python_commit"),
        "dest_resolved": str(dest),
    }
    return baseline_payload(records=records, n_jobs=n_jobs_i, extra=extra)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--scratch-out", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--tracked-out", type=Path, default=DEFAULT_TRACKED)
    parser.add_argument(
        "--write-tracked",
        action="store_true",
        help="Also write docs/reference/core/phase2-profiling-baseline.json",
    )
    args = parser.parse_args(argv)
    refuse_protected_stretch_energy_dest(args.scratch_out)
    payload = run_phase2_profiling_baseline(dest=args.dest)
    args.scratch_out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.scratch_out.write_text(text, encoding="utf-8")
    if args.write_tracked and payload.get("result") == "ok":
        args.tracked_out.parent.mkdir(parents=True, exist_ok=True)
        args.tracked_out.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "result": payload.get("result"),
                "bottleneck_measured_on_dest": payload.get("bottleneck_measured_on_dest"),
                "n_jobs": payload.get("n_jobs"),
                "scratch": str(args.scratch_out),
            },
            indent=2,
        )
    )
    if payload.get("result") == "ok":
        return 0
    if payload.get("result") == "skip":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
