"""Unit tests for MonitorDaemon."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from slavv_python.analytics.parity.monitor_daemon import MonitorDaemon


@pytest.fixture
def mock_registry():
    """Mock JobRegistry for testing."""
    with patch("slavv_python.analytics.parity.monitor_daemon.JobRegistry") as mock:
        yield mock.return_value


@pytest.fixture
def mock_process_utils():
    """Mock process utilities."""
    with patch("slavv_python.analytics.parity.monitor_daemon.is_process_alive") as alive_mock, patch(
        "slavv_python.analytics.parity.monitor_daemon.is_python_process"
    ) as python_mock:
        yield {"is_alive": alive_mock, "is_python": python_mock}


@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace directory."""
    workspace = tmp_path / "workspace" / "scratch"
    workspace.mkdir(parents=True)
    
    # Patch Path to use temp workspace
    original_path = Path
    
    def mock_path_new(path_str):
        if path_str == "workspace/scratch":
            return workspace
        return original_path(path_str)
    
    with patch("slavv_python.analytics.parity.monitor_daemon.Path", side_effect=mock_path_new):
        yield workspace


class TestMonitorDaemon:
    """Test MonitorDaemon class."""

    def test_initialization(self):
        """Test daemon initialization."""
        daemon = MonitorDaemon(poll_interval=15, idle_timeout_minutes=30)
        assert daemon.poll_interval == 15
        assert daemon.idle_timeout.total_seconds() == 1800

    def test_init_notifier_success(self):
        """Test notification library initialization."""
        with patch("slavv_python.analytics.parity.monitor_daemon.ToastNotifier"):
            daemon = MonitorDaemon(notification_enabled=True)
            assert daemon.notifier is not None

    def test_init_notifier_import_error(self):
        """Test graceful handling when notification library unavailable."""
        with patch(
            "slavv_python.analytics.parity.monitor_daemon.ToastNotifier",
            side_effect=ImportError,
        ):
            daemon = MonitorDaemon(notification_enabled=True)
            assert daemon.notifier is None

    def test_init_notifier_disabled(self):
        """Test notifications disabled."""
        daemon = MonitorDaemon(notification_enabled=False)
        assert daemon.notifier is None

    @patch("slavv_python.analytics.parity.monitor_daemon.is_process_alive")
    @patch("slavv_python.analytics.parity.monitor_daemon.read_daemon_pid")
    def test_is_running_when_daemon_alive(self, mock_read_pid, mock_is_alive):
        """Test is_running returns True when daemon process exists."""
        mock_read_pid.return_value = 12345
        mock_is_alive.return_value = True

        daemon = MonitorDaemon()
        assert daemon.is_running() is True

    @patch("slavv_python.analytics.parity.monitor_daemon.is_process_alive")
    @patch("slavv_python.analytics.parity.monitor_daemon.read_daemon_pid")
    def test_is_running_when_no_pid_file(self, mock_read_pid, mock_is_alive):
        """Test is_running returns False when no PID file."""
        mock_read_pid.return_value = None

        daemon = MonitorDaemon()
        assert daemon.is_running() is False

    @patch("slavv_python.analytics.parity.monitor_daemon.is_process_alive")
    @patch("slavv_python.analytics.parity.monitor_daemon.read_daemon_pid")
    def test_is_running_when_daemon_dead(self, mock_read_pid, mock_is_alive):
        """Test is_running returns False when daemon process is dead."""
        mock_read_pid.return_value = 12345
        mock_is_alive.return_value = False

        daemon = MonitorDaemon()
        assert daemon.is_running() is False

    @patch("slavv_python.analytics.parity.monitor_daemon.subprocess.Popen")
    @patch("slavv_python.analytics.parity.monitor_daemon.sys")
    @patch("slavv_python.analytics.parity.monitor_daemon.time.sleep")
    @patch("slavv_python.analytics.parity.monitor_daemon.is_process_alive")
    def test_start_daemon_success(self, mock_is_alive, mock_sleep, mock_sys, mock_popen):
        """Test successful daemon start."""
        mock_sys.executable = "/usr/bin/python"
        mock_sys.platform = "linux"
        mock_proc = Mock()
        mock_proc.pid = 99999
        mock_popen.return_value = mock_proc
        mock_is_alive.return_value = True

        daemon = MonitorDaemon()
        with patch.object(daemon, "is_running", return_value=False):
            result = daemon.start()

        assert result is True
        mock_popen.assert_called_once()

    @patch("slavv_python.analytics.parity.monitor_daemon.subprocess.Popen")
    def test_start_daemon_already_running(self, mock_popen):
        """Test start when daemon already running."""
        daemon = MonitorDaemon()
        with patch.object(daemon, "is_running", return_value=True):
            result = daemon.start()

        assert result is True
        mock_popen.assert_not_called()

    @patch("slavv_python.analytics.parity.monitor_daemon.kill_process_tree")
    @patch("slavv_python.analytics.parity.monitor_daemon.clear_daemon_pid")
    @patch("slavv_python.analytics.parity.monitor_daemon.read_daemon_pid")
    def test_stop_daemon(self, mock_read_pid, mock_clear_pid, mock_kill):
        """Test stopping daemon."""
        mock_read_pid.return_value = 12345
        mock_kill.return_value = True

        daemon = MonitorDaemon()
        result = daemon.stop()

        assert result is True
        mock_kill.assert_called_once_with(12345)
        mock_clear_pid.assert_called_once()

    @patch("slavv_python.analytics.parity.monitor_daemon.read_daemon_pid")
    def test_stop_daemon_not_running(self, mock_read_pid):
        """Test stop when daemon not running."""
        mock_read_pid.return_value = None

        daemon = MonitorDaemon()
        result = daemon.stop()

        assert result is True

    def test_format_duration(self):
        """Test duration formatting."""
        from datetime import timedelta

        daemon = MonitorDaemon()

        # Test hours and minutes
        duration = timedelta(hours=2, minutes=30)
        assert daemon._format_duration(duration) == "2h 30m"

        # Test minutes only
        duration = timedelta(minutes=45)
        assert daemon._format_duration(duration) == "45m"

        # Test zero duration
        duration = timedelta(seconds=0)
        assert daemon._format_duration(duration) == "0m"

    def test_write_heartbeat(self, tmp_path):
        """Test heartbeat file writing."""
        daemon = MonitorDaemon()
        daemon.heartbeat_file = tmp_path / "heartbeat.json"

        daemon._write_heartbeat()

        assert daemon.heartbeat_file.exists()
        data = json.loads(daemon.heartbeat_file.read_text())
        assert "timestamp" in data
        assert "pid" in data

    def test_send_notification_with_notifier(self):
        """Test sending notification when notifier available."""
        mock_notifier = Mock()
        daemon = MonitorDaemon()
        daemon.notifier = mock_notifier

        mock_job = Mock()
        mock_job.started_at = datetime.now().isoformat()
        mock_job.stage = "energy"
        mock_job.pid = 12345

        daemon._send_notification(mock_job, "completed", 0)

        mock_notifier.show_toast.assert_called_once()

    def test_send_notification_without_notifier(self):
        """Test notification logging when notifier unavailable."""
        daemon = MonitorDaemon()
        daemon.notifier = None

        mock_job = Mock()
        mock_job.started_at = datetime.now().isoformat()
        mock_job.stage = "energy"
        mock_job.pid = 12345

        # Should not raise error
        daemon._send_notification(mock_job, "completed", 0)

    def test_send_notification_failed(self):
        """Test graceful handling of notification failure."""
        mock_notifier = Mock()
        mock_notifier.show_toast.side_effect = Exception("Notification failed")

        daemon = MonitorDaemon()
        daemon.notifier = mock_notifier

        mock_job = Mock()
        mock_job.started_at = datetime.now().isoformat()
        mock_job.stage = "energy"
        mock_job.pid = 12345

        # Should not raise error
        daemon._send_notification(mock_job, "failed", 1)

    def test_get_exit_code(self, tmp_path):
        """Test reading exit code from job metadata."""
        daemon = MonitorDaemon()

        mock_job = Mock()
        mock_job.run_dir = str(tmp_path)

        # Create metadata file
        metadata_dir = tmp_path / "99_Metadata"
        metadata_dir.mkdir()
        metadata_file = metadata_dir / "parity_job.json"
        metadata_file.write_text(json.dumps({"exit_code": 42}))

        exit_code = daemon._get_exit_code(mock_job)
        assert exit_code == 42

    def test_get_exit_code_no_file(self, tmp_path):
        """Test getting exit code when metadata file doesn't exist."""
        daemon = MonitorDaemon()

        mock_job = Mock()
        mock_job.run_dir = str(tmp_path)

        exit_code = daemon._get_exit_code(mock_job)
        assert exit_code is None


@pytest.mark.unit
class TestMonitorDaemonRunLoop:
    """Test the daemon run loop logic."""

    @patch("slavv_python.analytics.parity.monitor_daemon.time.sleep")
    @patch("slavv_python.analytics.parity.monitor_daemon.signal")
    @patch("slavv_python.analytics.parity.monitor_daemon.os.getpid")
    @patch("slavv_python.analytics.parity.monitor_daemon.write_daemon_pid")
    @patch("slavv_python.analytics.parity.monitor_daemon.JobRegistry")
    @patch("slavv_python.analytics.parity.monitor_daemon.is_process_alive")
    @patch("slavv_python.analytics.parity.monitor_daemon.is_python_process")
    def test_run_loop_with_active_jobs(
        self,
        mock_is_python,
        mock_is_alive,
        mock_registry_class,
        mock_write_pid,
        mock_getpid,
        mock_signal,
        mock_sleep,
        tmp_path,
    ):
        """Test daemon run loop with active jobs."""
        mock_getpid.return_value = 99999
        mock_is_alive.return_value = True
        mock_is_python.return_value = True

        # Create mock job
        mock_job = Mock()
        mock_job.job_id = "test-job"
        mock_job.pid = 12345
        mock_job.started_at = datetime.now().isoformat()

        # Mock registry to return one active job, then empty
        mock_registry = Mock()
        mock_registry.get_active_jobs.side_effect = [[mock_job], []]
        mock_registry_class.return_value = mock_registry

        daemon = MonitorDaemon(idle_timeout_minutes=0)  # Immediate timeout
        daemon.log_file = tmp_path / "daemon.log"
        daemon.heartbeat_file = tmp_path / "heartbeat.json"
        daemon.pid_file = tmp_path / "daemon.pid"

        # Run loop should exit after idle timeout
        daemon.run()

        # Verify job was checked
        assert mock_registry.update_job.called

    @patch("slavv_python.analytics.parity.monitor_daemon.time.sleep")
    @patch("slavv_python.analytics.parity.monitor_daemon.signal")
    @patch("slavv_python.analytics.parity.monitor_daemon.os.getpid")
    @patch("slavv_python.analytics.parity.monitor_daemon.write_daemon_pid")
    @patch("slavv_python.analytics.parity.monitor_daemon.JobRegistry")
    def test_run_loop_idle_shutdown(
        self, mock_registry_class, mock_write_pid, mock_getpid, mock_signal, mock_sleep, tmp_path
    ):
        """Test daemon shuts down when idle."""
        mock_getpid.return_value = 99999

        # Mock registry to always return empty
        mock_registry = Mock()
        mock_registry.get_active_jobs.return_value = []
        mock_registry_class.return_value = mock_registry

        daemon = MonitorDaemon(idle_timeout_minutes=0, poll_interval=0.1)
        daemon.log_file = tmp_path / "daemon.log"
        daemon.heartbeat_file = tmp_path / "heartbeat.json"
        daemon.pid_file = tmp_path / "daemon.pid"

        # Should exit immediately due to idle timeout
        daemon.run()

        assert mock_registry.get_active_jobs.called
