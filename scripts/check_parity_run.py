"""One-shot health check for an exact-route parity run.

Reads a run's own authoritative signals (no external harness log required) and
prints a single verdict -- RUNNING / STALLED / COMPLETED / FAILED / PENDING --
plus per-stage progress and liveness. Liveness uses the resume/snapshot
heartbeat AGE (reliable); it does NOT trust run-dir start-time clocks, which can
be stale across restarts. It deliberately omits an ETA: the only run-dir
progress signal is the merge cursor, which lags the parallel compute under
n_jobs>1 -- use scripts/parity_run_throughput.py --log for the real chunk rate.

Usage::

    python scripts/check_parity_run.py --run-dir workspace/runs/oracle_180709_E/canonical_full_v3
    python scripts/check_parity_run.py --run-dir <run> --stall-min 20
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

_STAGES = ("preprocess", "energy", "vertices", "edges", "network")


def _load(path: Path) -> dict:
    """Load a JSON dict, or {} if absent/unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _age_seconds(iso: str | None, now: datetime) -> float | None:
    """Seconds between an ISO-8601 (Z) timestamp and now, or None."""
    if not iso:
        return None
    try:
        ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (now - ts).total_seconds()


def _fmt_age(secs: float | None) -> str:
    """Human-readable age."""
    if secs is None:
        return "unknown"
    if secs < 90:
        return f"{secs:.0f}s ago"
    if secs < 5400:
        return f"{secs / 60:.0f}m ago"
    return f"{secs / 3600:.1f}h ago"


def main() -> None:
    """Print a verdict + progress + liveness for an exact-route parity run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stall-min", type=float, default=15.0, help="Heartbeat age (min) to flag STALLED")
    args = parser.parse_args()
    run = args.run_dir
    now = datetime.now(timezone.utc)

    snapshot = _load(run / "99_Metadata/run_snapshot.json")
    job = _load(run / "99_Metadata/parity_job.json")
    resume = _load(run / "02_Energy/resume_state.json")
    ckpt_dir = run / "02_Output/python_results/checkpoints"
    checkpoints = {p.stem.replace("checkpoint_", "") for p in ckpt_dir.glob("checkpoint_*.pkl")} if ckpt_dir.is_dir() else set()

    stages = snapshot.get("stages") or {}
    # Freshest heartbeat across the live signals (run-dir clocks; ages are reliable).
    ages = [
        _age_seconds(resume.get("heartbeat_at"), now),
        _age_seconds(snapshot.get("updated_at"), now),
    ]
    heartbeat_age = min((a for a in ages if a is not None), default=None)

    # Verdict.
    job_status = str(job.get("status") or snapshot.get("status") or "unknown")
    exit_code = job.get("exit_code")
    all_done = all(s in checkpoints for s in ("energy", "vertices", "edges", "network"))
    if exit_code not in (None, 0) or job.get("reason"):
        verdict = "FAILED"
    elif job_status in {"completed", "succeeded"} or (all_done and snapshot.get("status") == "completed"):
        verdict = "COMPLETED"
    elif job_status == "running" and heartbeat_age is not None and heartbeat_age > args.stall_min * 60:
        verdict = "STALLED"
    elif job_status == "running":
        verdict = "RUNNING"
    else:
        verdict = job_status.upper()

    print(f"RUN: {run.name}")
    print(f"VERDICT: {verdict}   (heartbeat {_fmt_age(heartbeat_age)})")
    if verdict == "FAILED":
        print(f"  exit_code={exit_code}  reason={job.get('reason')}")

    # Per-stage line.
    done = [s for s in _STAGES if s in checkpoints]
    cur = snapshot.get("current_stage", "?")
    parts = []
    for s in _STAGES:
        st = stages.get(s, {})
        mark = "[x]" if s in checkpoints else {"running": "[>]", "completed": "[x]"}.get(str(st.get("status")), "[ ]")
        parts.append(f"{mark}{s}")
    print("stages: " + "  ".join(parts) + f"   (current: {cur}, checkpoints: {len(done)}/4)")

    # Energy octave detail + overall progress.
    # NOTE: under n_jobs>1 these are the MERGE cursor, which lags the parallel
    # compute — the joblib "Done N tasks" log is the leading indicator. Use
    # scripts/parity_run_throughput.py --log <run-log> for the live chunk rate.
    if resume.get("units_total"):
        u, t = int(resume.get("completed_units", 0)), int(resume["units_total"])
        print(f"energy octave {resume.get('octave')}: merge cursor {u}/{t} ({100 * u / t:.1f}%) -- lags compute under n_jobs>1")
    overall = snapshot.get("overall_progress")
    if overall is not None:
        print(f"overall progress: {100 * float(overall):.1f}% (merge cursor; lags compute)")

    # Deliberately no ETA here: the only run-dir progress signal (the merge
    # cursor) lags the parallel compute, so an ETA from it is garbage. The
    # reliable rate is the joblib "Done N tasks" timeline in the run log.
    print("rate/ETA: scripts/parity_run_throughput.py --log <run-log> --total-chunks <N>")


if __name__ == "__main__":
    main()
