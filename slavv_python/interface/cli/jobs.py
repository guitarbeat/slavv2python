"""CLI commands for parity job monitoring."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from slavv_python.analytics.parity.runs.job_registry import JobRegistry
from slavv_python.analytics.parity.runs.monitor_daemon import MonitorDaemon
from slavv_python.analytics.parity.runs.process_utils import (
    is_process_alive,
    kill_process_tree,
    read_daemon_pid,
)


def format_duration(duration: timedelta) -> str:
    """Format duration as human-readable string."""
    total_seconds = int(duration.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def truncate_path(path_str: str, max_len: int = 30) -> str:
    """Truncate path to fit in table."""
    if len(path_str) <= max_len:
        return path_str

    # Show last part
    parts = Path(path_str).parts
    if len(parts) > 1:
        return "..." + str(Path(*parts[-2:]))
    return path_str[-max_len:]


def cmd_list(args: argparse.Namespace) -> None:
    """List active parity jobs."""
    try:
        from tabulate import tabulate
    except ImportError:
        print("Error: tabulate library not installed", file=sys.stderr)
        print("Install with: pip install tabulate", file=sys.stderr)
        sys.exit(1)

    registry = JobRegistry()
    jobs = registry.get_active_jobs()

    if not jobs:
        print("No active parity jobs")
        return

    print("\nActive Parity Jobs:")
    rows = []
    for job in jobs:
        started = datetime.fromisoformat(job.started_at)
        duration = datetime.now() - started

        rows.append(
            [
                job.job_id[:8],
                job.pid,
                truncate_path(job.run_dir),
                job.stage,
                job.status,
                started.strftime("%Y-%m-%d %H:%M"),
                format_duration(duration),
            ]
        )

    print(
        tabulate(
            rows,
            headers=["Job ID", "PID", "Run Directory", "Stage", "Status", "Started", "Duration"],
            tablefmt="simple",
        )
    )
    print()


def cmd_history(args: argparse.Namespace) -> None:
    """Show job history."""
    try:
        from tabulate import tabulate
    except ImportError:
        print("Error: tabulate library not installed", file=sys.stderr)
        print("Install with: pip install tabulate", file=sys.stderr)
        sys.exit(1)

    registry = JobRegistry()
    run_dir = Path(args.run_dir) if args.run_dir else None
    limit = int(args.limit) if hasattr(args, "limit") and args.limit else None

    jobs = registry.get_job_history(run_dir=run_dir, limit=limit)

    if not jobs:
        if run_dir:
            print(f"No job history for: {run_dir}")
        else:
            print("No job history")
        return

    if run_dir:
        print(f"\nJob History for: {run_dir}")
    else:
        print("\nJob History:")

    rows = []
    for job in jobs:
        started = datetime.fromisoformat(job.started_at)

        if job.completed_at:
            completed = datetime.fromisoformat(job.completed_at)
            duration = completed - started
        else:
            duration = datetime.now() - started

        rows.append(
            [
                job.job_id[:8],
                job.pid,
                started.strftime("%Y-%m-%d %H:%M"),
                job.stage,
                job.exit_code if job.exit_code is not None else "-",
                format_duration(duration),
                job.status,
            ]
        )

    print(
        tabulate(
            rows,
            headers=["Job ID", "PID", "Started", "Stage", "Exit", "Duration", "Status"],
            tablefmt="simple",
        )
    )
    print()


def cmd_kill(args: argparse.Namespace) -> None:
    """Kill a running job."""
    registry = JobRegistry()
    job_id = args.job_id

    # Find job (accept partial IDs)
    jobs = registry.get_active_jobs()
    matching = [j for j in jobs if j.job_id.startswith(job_id)]

    if not matching:
        print(f"Error: No active job found matching '{job_id}'", file=sys.stderr)
        sys.exit(1)

    if len(matching) > 1:
        print(f"Error: Multiple jobs match '{job_id}':", file=sys.stderr)
        for j in matching:
            print(f"  {j.job_id} (PID {j.pid})", file=sys.stderr)
        sys.exit(1)

    job = matching[0]

    if not is_process_alive(job.pid):
        print(f"Job {job.job_id[:8]} (PID {job.pid}) is not running")
        registry.update_job(job.job_id, status="completed", completed_at=datetime.now().isoformat())
        return

    print(f"Terminating job {job.job_id[:8]} (PID {job.pid})...")
    success = kill_process_tree(job.pid)

    if success:
        registry.update_job(
            job.job_id, status="killed", completed_at=datetime.now().isoformat(), exit_code=-1
        )
        print("Job terminated successfully")
    else:
        print("Failed to terminate job", file=sys.stderr)
        sys.exit(1)


def cmd_daemon_status(args: argparse.Namespace) -> None:
    """Show daemon status."""
    MonitorDaemon()

    pid = read_daemon_pid()
    if pid is None:
        print("Daemon: Not running (no PID file)")
        return

    alive = is_process_alive(pid)
    if not alive:
        print(f"Daemon: Stopped (stale PID {pid})")
        return

    # Read heartbeat
    heartbeat_file = Path("workspace/scratch/monitor_daemon_heartbeat.json")
    if heartbeat_file.exists():
        import json

        try:
            heartbeat = json.loads(heartbeat_file.read_text())
            last_beat = datetime.fromisoformat(heartbeat["timestamp"])
            uptime = datetime.now() - last_beat
            print(f"Daemon: Running (PID {pid})")
            print(f"Last heartbeat: {last_beat.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Age: {format_duration(uptime)}")
        except Exception as e:
            print(f"Daemon: Running (PID {pid})")
            print(f"Warning: Could not read heartbeat: {e}")
    else:
        print(f"Daemon: Running (PID {pid})")
        print("No heartbeat file found")

    # Count active jobs
    registry = JobRegistry()
    active_jobs = registry.get_active_jobs()
    print(f"Active jobs: {len(active_jobs)}")


def cmd_daemon_restart(args: argparse.Namespace) -> None:
    """Restart the daemon."""
    daemon = MonitorDaemon()

    print("Stopping daemon...")
    daemon.stop()

    print("Starting daemon...")
    success = daemon.start()

    if success:
        print("Daemon restarted successfully")
    else:
        print("Failed to start daemon", file=sys.stderr)
        sys.exit(1)


def build_jobs_parser() -> argparse.ArgumentParser:
    """Build the jobs CLI parser."""
    parser = argparse.ArgumentParser(
        prog="slavv jobs", description="Manage and monitor parity jobs"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # List command
    parser_list = subparsers.add_parser("list", help="List active jobs")
    parser_list.set_defaults(handler=cmd_list)

    # History command
    parser_history = subparsers.add_parser("history", help="Show job history")
    parser_history.add_argument("--run-dir", help="Filter by run directory")
    parser_history.add_argument("--limit", type=int, help="Maximum number of jobs to show")
    parser_history.set_defaults(handler=cmd_history)

    # Kill command
    parser_kill = subparsers.add_parser("kill", help="Terminate a running job")
    parser_kill.add_argument("job_id", help="Job ID (or prefix) to kill")
    parser_kill.set_defaults(handler=cmd_kill)

    # Daemon subcommand
    parser_daemon = subparsers.add_parser("daemon", help="Manage monitoring daemon")
    daemon_subparsers = parser_daemon.add_subparsers(dest="daemon_cmd", required=True)

    # Daemon status
    parser_daemon_status = daemon_subparsers.add_parser("status", help="Show daemon status")
    parser_daemon_status.set_defaults(handler=cmd_daemon_status)

    # Daemon restart
    parser_daemon_restart = daemon_subparsers.add_parser("restart", help="Restart daemon")
    parser_daemon_restart.set_defaults(handler=cmd_daemon_restart)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main entry point for jobs CLI."""
    parser = build_jobs_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
