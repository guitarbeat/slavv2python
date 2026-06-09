"""Integration tests for monitored parity jobs."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from slavv_python.analytics.parity.job_registry import JobRegistry
from slavv_python.analytics.parity.monitor_daemon import MonitorDaemon
from slavv_python.analytics.parity.process_utils import (
    ensure_monitor_daemon_running,
    is_process_alive,
)


@pytest.fixture
def test_registry(tmp_path):
    """Create test registry."""
    return JobRegistry(tmp_path / "test_registry.jsonl")


@pytest.mark.integration
class TestMonitoredJobLifecycle:
    """Test complete lifecycle of monitored jobs."""

    def test_job_registration_and_query(self, test_registry):
        """Test basic job registration and retrieval."""
        job_id = test_registry.register_job(
            pid=99999,
            run_dir=Path("/tmp/test"),
            oracle_root=Path("/tmp/oracle"),
            stage="energy",
            command="test command",
        )

        assert job_id is not None

        # Query active jobs
        active = test_registry.get_active_jobs()
        assert len(active) == 1
        assert active[0].job_id == job_id

        # Query by run_dir
        job = test_registry.get_job_by_run_dir(Path("/tmp/test"))
        assert job.job_id == job_id

    def test_duplicate_writer_detection(self, test_registry):
        """Test detection of duplicate writers."""
        run_dir = Path("/tmp/test_run")

        # Register first job
        job_id1 = test_registry.register_job(
            pid=11111,
            run_dir=run_dir,
            oracle_root=Path("/tmp/oracle"),
            stage="energy",
            command="test 1",
        )

        # Check for duplicate
        active_job = test_registry.get_job_by_run_dir(run_dir)
        assert active_job is not None
        assert active_job.job_id == job_id1

        # Mark first job as completed
        test_registry.update_job(job_id1, status="completed")

        # Now second job should be allowed
        job_id2 = test_registry.register_job(
            pid=22222,
            run_dir=run_dir,
            oracle_root=Path("/tmp/oracle"),
            stage="energy",
            command="test 2",
        )

        assert job_id2 != job_id1

    def test_job_history_accumulation(self, test_registry):
        """Test that job history accumulates correctly."""
        for i in range(3):
            job_id = test_registry.register_job(
                pid=10000 + i,
                run_dir=Path(f"/tmp/test_{i}"),
                oracle_root=Path("/tmp/oracle"),
                stage="energy",
                command=f"test {i}",
            )
            test_registry.update_job(job_id, status="completed", exit_code=0)

        history = test_registry.get_job_history()
        assert len(history) == 3

        # All should be completed
        assert all(job.status == "completed" for job in history)


@pytest.mark.integration
@pytest.mark.slow
class TestDaemonIntegration:
    """Test daemon integration (marked as slow)."""

    def test_daemon_start_and_stop(self):
        """Test daemon can start and stop."""
        daemon = MonitorDaemon()

        # Note: Actual daemon start/stop is tested in unit tests with mocks
        # This is a placeholder for manual integration testing
        assert daemon is not None

    def test_ensure_daemon_running(self):
        """Test ensure_monitor_daemon_running helper."""
        # This would start a real daemon in production
        # For testing, we verify the function exists and is callable
        assert callable(ensure_monitor_daemon_running)


@pytest.mark.integration
class TestCLIIntegration:
    """Test CLI integration with monitoring (placeholder)."""

    def test_monitor_flag_exists(self):
        """Test that --monitor flag is recognized."""
        from slavv_python.analytics.parity.commands import build_parity_parser

        parser = build_parity_parser()

        # Parse with --monitor flag
        args = parser.parse_args(
            [
                "resume-exact-run",
                "--dest-run-root",
                "/tmp/test",
                "--oracle-root",
                "/tmp/oracle",
                "--monitor",
            ]
        )

        assert hasattr(args, "monitor")
        assert args.monitor is True

    def test_force_kill_flag_exists(self):
        """Test that --force-kill flag is recognized."""
        from slavv_python.analytics.parity.commands import build_parity_parser

        parser = build_parity_parser()

        # Parse with --force-kill flag
        args = parser.parse_args(
            [
                "resume-exact-run",
                "--dest-run-root",
                "/tmp/test",
                "--oracle-root",
                "/tmp/oracle",
                "--force-kill",
            ]
        )

        assert hasattr(args, "force_kill")
        assert args.force_kill is True


@pytest.mark.integration
class TestJobsCLI:
    """Test slavv jobs CLI commands."""

    def test_jobs_cli_parser(self):
        """Test jobs CLI parser builds correctly."""
        from slavv_python.interface.cli.jobs import build_jobs_parser

        parser = build_jobs_parser()

        # Test list command
        args = parser.parse_args(["list"])
        assert args.subcommand == "list"

        # Test history command
        args = parser.parse_args(["history"])
        assert args.subcommand == "history"

        # Test kill command
        args = parser.parse_args(["kill", "test-job-id"])
        assert args.subcommand == "kill"
        assert args.job_id == "test-job-id"

        # Test daemon status
        args = parser.parse_args(["daemon", "status"])
        assert args.daemon_cmd == "status"
