"""Print pipeline progress from run_snapshot and energy resume_state."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from slavv_python.analytics.parity.constants import CHECKPOINTS_DIR, RUN_SNAPSHOT_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor SLAVV run progress.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    snapshot_path = run_dir / RUN_SNAPSHOT_PATH
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"missing snapshot: {snapshot_path}")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    print(f"run_id={snapshot.get('run_id')} status={snapshot.get('status')}")
    print(f"current_stage={snapshot.get('current_stage')} detail={snapshot.get('current_detail')}")
    print(f"overall_progress={snapshot.get('overall_progress')}")

    energy = (snapshot.get("stages") or {}).get("energy") or {}
    units_total = int(energy.get("units_total") or 0)
    units_completed = int(energy.get("units_completed") or 0)
    if units_total:
        print(f"energy_units={units_completed}/{units_total} ({100.0 * units_completed / units_total:.1f}%)")

    resume_path = run_dir / "02_Energy" / "resume_state.json"
    if resume_path.is_file():
        resume = json.loads(resume_path.read_text(encoding="utf-8"))
        completed = len(resume.get("completed_units") or [])
        print(f"energy_resume_completed_units={completed}")

    checkpoints = run_dir / CHECKPOINTS_DIR
    if checkpoints.is_dir():
        names = sorted(path.name for path in checkpoints.glob("checkpoint_*.pkl"))
        if names:
            print("checkpoints:", ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
