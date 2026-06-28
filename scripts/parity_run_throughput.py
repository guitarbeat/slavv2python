"""Report progress and throughput for a parity exact-run.

Reads a run's ``run_snapshot.json`` for the active stage, and (optionally) parses
the background log for joblib ``Done N tasks | elapsed`` lines to compute the
real chunk throughput and ETA. With ``n_jobs>1`` the per-chunk progress callback
only fires after each octave's parallel batch, so the joblib log is the most
reliable live signal for the energy stage.

Usage::

    python scripts/parity_run_throughput.py --run-dir <run> [--log <logfile>]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_DONE_RE = re.compile(r"Done\s+(\d+)\s+tasks.*?elapsed:\s+([0-9.]+)\s*(min|s)")


def _elapsed_seconds(value: float, unit: str) -> float:
    """Convert a joblib elapsed value to seconds."""
    return value * 60.0 if unit == "min" else value


def _read_snapshot(run_dir: Path) -> dict[str, object]:
    """Return the run snapshot dict, or an empty dict if absent/unreadable."""
    path = run_dir / "99_Metadata" / "run_snapshot.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _parse_joblib_progress(log_path: Path) -> list[tuple[int, float]]:
    """Return (tasks_done, elapsed_seconds) points from joblib log lines."""
    if not log_path.is_file():
        return []
    points: list[tuple[int, float]] = []
    for match in _DONE_RE.finditer(log_path.read_text(encoding="utf-8", errors="ignore")):
        points.append(
            (int(match.group(1)), _elapsed_seconds(float(match.group(2)), match.group(3)))
        )
    return points


def main() -> None:
    """Print a concise progress + throughput summary for a parity run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Run root directory")
    parser.add_argument("--log", type=Path, default=None, help="Background log file to parse")
    parser.add_argument(
        "--total-chunks",
        type=int,
        default=None,
        help="Total chunks in the current octave, for ETA (else inferred from resume_state).",
    )
    args = parser.parse_args()

    snapshot = _read_snapshot(args.run_dir)
    stage = snapshot.get("current_stage", "?")
    detail = snapshot.get("current_detail", "")
    print(f"stage: {stage}  | {detail}")

    total = args.total_chunks
    if total is None:
        resume = args.run_dir / "02_Energy" / "resume_state.json"
        if resume.is_file():
            try:
                total = int(json.loads(resume.read_text(encoding="utf-8")).get("units_total") or 0)
            except (OSError, ValueError):
                total = None

    if args.log is None:
        print("(no --log given; pass the background log for throughput)")
        return

    points = _parse_joblib_progress(args.log)
    if len(points) < 2:
        print("(not enough joblib progress lines yet)")
        return

    (n0, t0), (n1, t1) = points[-2], points[-1]
    if n1 <= n0 or t1 <= t0:
        print(f"done {n1} tasks @ {t1:.0f}s (rate unavailable)")
        return

    per_chunk = (t1 - t0) / (n1 - n0)
    print(f"done {n1} chunks @ {t1 / 60:.1f} min  | ~{per_chunk:.1f}s/chunk")
    if total:
        remaining = max(0, total - n1)
        eta_h = remaining * per_chunk / 3600.0
        print(f"octave total {total} chunks  | remaining {remaining}  | ETA ~{eta_h:.1f} h")


if __name__ == "__main__":
    main()
