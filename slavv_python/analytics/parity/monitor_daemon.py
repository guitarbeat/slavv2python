"""Background monitoring daemon for parity jobs."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class MonitorDaemon:
    """Background daemon that monitors active parity jobs."""

    def __init__(
        self,
        poll_interval: int = 30,
        idle_timeout_minutes: int = 60,
        notification_enabled: bool = True,
    ):
        """
        Initialize monitor daemon.

        Args:
            poll_interval: Seconds between polling cycles
            poll_interval: Seconds between status checks
            idle_timeout_minutes: Minutes of no active jobs before auto-shutdown
            notification_enabled: Whether to send desktop notifications
        """
        self.poll_interval = poll_interval
        self.idle_timeout = timedelta(minutes=idle_timeout_minutes)
        self.notification_enabled = notification_enabled

        workspace = Path("workspace/scratch")
        workspace.mkdir(parents=True, exist_ok=True)

        self.pid_file = workspace / "monitor_daemon.pid"
        self.log_file = workspace / "monitor_daemon.log"
        self.heartbeat_file = workspace / "monitor_daemon_heartbeat.json"

        # Try to import notification library
        self.notifier = self._init_notifier()

    def _init_notifier(self):
        """Initialize desktop notification library."""
        if not self.notification_enabled:
            return None

        try:
            from win10toast import ToastNotifier

            return ToastNotifier()
        except ImportError:
            logger.warning("win10toast not available, notifications disabled")
            return None

    def is_running(self) -> bool:
        """
        Check if daemon is currently running.

        Returns:
            True if daemon process is alive
        """
        from slavv_python.analytics.parity.process_utils import (
            is_process_alive,
            read_daemon_pid,
        )

        pid = read_daemon_pid()
        if pid is None:
            return False

        return is_process_alive(pid)

    def start(self) -> bool:
        """
        Start the daemon as a detached background process.

        Returns:
            True if daemon started successfully
        """
        if self.is_running():
            logger.info("Daemon already running")
            return True

        # Start detached process
        python_exe = sys.executable
        Path(__file__).resolve()

        # Build command to run daemon
        cmd = [python_exe, "-m", "slavv_python.analytics.parity.monitor_daemon"]

        try:
            # Platform-specific detached process creation
            if sys.platform == "win32":
                # Windows: DETACHED_PROCESS flag
                DETACHED_PROCESS = 0x00000008
                proc = subprocess.Popen(
                    cmd,
                    creationflags=DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
            else:
                # Unix: double fork pattern
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )

            # Give it a moment to start
            time.sleep(0.5)

            from slavv_python.analytics.parity.process_utils import is_process_alive

            if is_process_alive(proc.pid):
                logger.info(f"Daemon started with PID {proc.pid}")
                return True
            logger.error("Daemon process died immediately after start")
            return False

        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
            return False

    def stop(self) -> bool:
        """
        Stop the running daemon.

        Returns:
            True if daemon was stopped
        """
        from slavv_python.analytics.parity.process_utils import (
            clear_daemon_pid,
            kill_process_tree,
            read_daemon_pid,
        )

        pid = read_daemon_pid()
        if pid is None:
            logger.info("Daemon not running")
            return True

        success = kill_process_tree(pid)
        if success:
            clear_daemon_pid()
            logger.info("Daemon stopped")

        return success

    def run(self) -> None:
        """
        Main daemon loop. This method blocks.

        Should be called as the entry point for the daemon process.
        """
        from slavv_python.analytics.parity.job_registry import JobRegistry
        from slavv_python.analytics.parity.process_utils import (
            write_daemon_pid,
        )

        # Setup logging to file
        self._setup_logging()

        # Write PID file
        write_daemon_pid(os.getpid())
        logger.info(f"Daemon started (PID {os.getpid()})")

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        registry = JobRegistry()
        last_active_time = datetime.now()
        self._running = True

        try:
            while self._running:
                # Update heartbeat
                self._write_heartbeat()

                # Get active jobs
                active_jobs = registry.get_active_jobs()

                if active_jobs:
                    last_active_time = datetime.now()
                    logger.debug(f"Monitoring {len(active_jobs)} active jobs")

                    # Check each job
                    for job in active_jobs:
                        self._check_job(job, registry)
                else:
                    # Check for idle timeout
                    idle_duration = datetime.now() - last_active_time
                    if idle_duration > self.idle_timeout:
                        logger.info(f"No active jobs for {idle_duration}, shutting down")
                        break

                # Sleep until next poll
                time.sleep(self.poll_interval)

        except Exception as e:
            logger.error(f"Daemon crashed: {e}", exc_info=True)
        finally:
            self._cleanup()

    def _check_job(self, job, registry) -> None:
        """Check status of a single job."""
        from slavv_python.analytics.parity.process_utils import (
            is_process_alive,
            is_python_process,
        )

        if is_process_alive(job.pid) and is_python_process(job.pid):
            # Job still running, update last_seen_at
            registry.update_job(job.job_id, last_seen_at=datetime.now().isoformat())
            logger.debug(f"Job {job.job_id} (PID {job.pid}) still running")
        else:
            # Job completed or failed
            self._handle_completed_job(job, registry)

    def _handle_completed_job(self, job, registry) -> None:
        """Handle a job that has completed."""
        logger.info(f"Job {job.job_id} (PID {job.pid}) has completed")

        # Try to determine exit code from run-local metadata
        exit_code = self._get_exit_code(job)
        if exit_code is None:
            status = "interrupted"
        elif exit_code == 0:
            status = "completed"
        else:
            status = "failed"

        reason = ""
        if status == "interrupted":
            reason = f"Writer PID {job.pid} is no longer alive."
            try:
                from slavv_python.analytics.parity.parity_job_lifecycle import (
                    reconcile_interrupted_run,
                )

                reconcile_interrupted_run(Path(job.run_dir), reason=reason)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Failed to reconcile interrupted parity job for %s: %s",
                    job.run_dir,
                    exc,
                )

        registry_status = "succeeded" if status == "completed" else status
        registry.update_job(
            job.job_id,
            status=registry_status,
            completed_at=datetime.now().isoformat(),
            exit_code=exit_code,
            metadata={"reason": reason} if reason else None,
        )

        # Send notification
        self._send_notification(job, status, exit_code)

    def _get_exit_code(self, job) -> int | None:
        """Try to get exit code from job metadata."""
        try:
            metadata_file = Path(job.run_dir) / "99_Metadata" / "parity_job.json"
            if metadata_file.exists():
                data = json.loads(metadata_file.read_text())
                return data.get("exit_code")
        except Exception as e:
            logger.debug(f"Could not read exit code for job {job.job_id}: {e}")

        return None

    def _send_notification(self, job, status: str, exit_code: int | None) -> None:
        """Send desktop notification for job completion."""
        # Calculate duration
        started = datetime.fromisoformat(job.started_at)
        duration = datetime.now() - started
        duration_str = self._format_duration(duration)

        # Build notification message
        title = f"Parity Job {'Completed' if status == 'completed' else 'Failed'}"
        message = (
            f"Stage: {job.stage}\n"
            f"Duration: {duration_str}\n"
            f"PID: {job.pid}\n"
            f"Exit code: {exit_code if exit_code is not None else 'unknown'}"
        )

        # Log the notification
        logger.info(f"Notification: {title} - {message}")

        # Try to send desktop notification
        if self.notifier:
            try:
                self.notifier.show_toast(
                    title,
                    message,
                    duration=10,
                    threaded=True,
                )
            except Exception as e:
                logger.warning(f"Failed to send desktop notification: {e}")

    def _format_duration(self, duration: timedelta) -> str:
        """Format duration as human-readable string."""
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60

        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def _write_heartbeat(self) -> None:
        """Write heartbeat file with current timestamp."""
        heartbeat = {
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid(),
        }
        try:
            self.heartbeat_file.write_text(json.dumps(heartbeat))
        except Exception as e:
            logger.warning(f"Failed to write heartbeat: {e}")

    def _setup_logging(self) -> None:
        """Configure logging to file."""
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Get root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.DEBUG)

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down")
        self._running = False

    def _cleanup(self) -> None:
        """Cleanup on shutdown."""
        from slavv_python.analytics.parity.process_utils import clear_daemon_pid

        clear_daemon_pid()
        logger.info("Daemon shutdown complete")


def main():
    """Entry point for daemon process."""

    daemon = MonitorDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
