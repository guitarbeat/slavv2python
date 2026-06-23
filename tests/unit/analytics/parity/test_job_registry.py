"""Unit tests for JobRegistry."""

from __future__ import annotations

import json
import multiprocessing
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from slavv_python.analytics.parity.job_registry import JobRegistry, ParityJobRecord


@pytest.fixture
def temp_registry(tmp_path):
    """Create a temporary registry for testing."""
    registry_path = tmp_path / "test_registry.jsonl"
    return JobRegistry(registry_path)


@pytest.fixture
def sample_job_data():
    """Sample job data for testing."""
    return {
        "pid": 12345,
        "run_dir": Path("/tmp/test_run"),
        "oracle_root": Path("/tmp/oracle"),
        "stage": "energy",
        "command": "test command",
        "metadata": {"test": "data"},
    }


class TestParityJobRecord:
    """Test ParityJobRecord dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = ParityJobRecord(
            job_id="test-id",
            pid=123,
            run_dir="/tmp/test",
            oracle_root="/tmp/oracle",
            stage="energy",
            command="test",
            started_at="2026-01-01T00:00:00",
            last_seen_at="2026-01-01T00:00:00",
        )
        data = record.to_dict()
        assert data["job_id"] == "test-id"
        assert data["pid"] == 123
        assert data["stage"] == "energy"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "job_id": "test-id",
            "pid": 123,
            "run_dir": "/tmp/test",
            "oracle_root": "/tmp/oracle",
            "stage": "energy",
            "command": "test",
            "started_at": "2026-01-01T00:00:00",
            "last_seen_at": "2026-01-01T00:00:00",
            "completed_at": None,
            "exit_code": None,
            "status": "running",
            "metadata": None,
        }
        record = ParityJobRecord.from_dict(data)
        assert record.job_id == "test-id"
        assert record.pid == 123
        assert record.status == "running"


class TestJobRegistry:
    """Test JobRegistry class."""

    def test_initialization(self, tmp_path):
        """Test registry initialization."""
        registry = JobRegistry(tmp_path / "registry.jsonl")
        assert registry.registry_path.exists()

    def test_register_job(self, temp_registry, sample_job_data):
        """Test job registration."""
        job_id = temp_registry.register_job(**sample_job_data)
        assert job_id is not None
        assert len(job_id) == 36  # UUID format

        # Verify job was written to file
        with open(temp_registry.registry_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["job_id"] == job_id
            assert data["pid"] == 12345

    def test_get_active_jobs(self, temp_registry, sample_job_data):
        """Test retrieving active jobs."""
        job_id = temp_registry.register_job(**sample_job_data)
        active_jobs = temp_registry.get_active_jobs()
        assert len(active_jobs) == 1
        assert active_jobs[0].job_id == job_id
        assert active_jobs[0].status == "running"

    def test_get_active_jobs_filters_completed(self, temp_registry, sample_job_data):
        """Test that completed jobs are not returned as active."""
        job_id = temp_registry.register_job(**sample_job_data)
        temp_registry.update_job(job_id, status="completed")

        active_jobs = temp_registry.get_active_jobs()
        assert len(active_jobs) == 0

    def test_update_job(self, temp_registry, sample_job_data):
        """Test job updates."""
        job_id = temp_registry.register_job(**sample_job_data)

        # Update job status
        temp_registry.update_job(
            job_id, status="completed", exit_code=0, completed_at=datetime.now().isoformat()
        )

        # Verify update
        job = temp_registry.get_job_by_id(job_id)
        assert job.status == "completed"
        assert job.exit_code == 0
        assert job.completed_at is not None

    def test_get_job_by_run_dir(self, temp_registry, sample_job_data):
        """Test finding job by run directory."""
        job_id = temp_registry.register_job(**sample_job_data)
        job = temp_registry.get_job_by_run_dir(sample_job_data["run_dir"])

        assert job is not None
        assert job.job_id == job_id
        assert str(Path(job.run_dir).resolve()) == str(Path(sample_job_data["run_dir"]).resolve())

    def test_get_job_by_run_dir_returns_latest(self, temp_registry, sample_job_data):
        """Test that get_job_by_run_dir returns the most recent job."""
        # Register two jobs for same run_dir
        job_id1 = temp_registry.register_job(**sample_job_data)
        temp_registry.update_job(job_id1, status="completed")

        job_id2 = temp_registry.register_job(**sample_job_data)

        job = temp_registry.get_job_by_run_dir(sample_job_data["run_dir"])
        assert job.job_id == job_id2  # Should return the more recent one

    def test_get_job_history(self, temp_registry, sample_job_data):
        """Test retrieving job history."""
        job_id1 = temp_registry.register_job(**sample_job_data)
        temp_registry.update_job(job_id1, status="completed")

        sample_job_data["run_dir"] = Path("/tmp/different_run")
        temp_registry.register_job(**sample_job_data)

        history = temp_registry.get_job_history()
        assert len(history) == 2

    def test_get_job_history_with_filter(self, temp_registry, sample_job_data):
        """Test retrieving job history filtered by run_dir."""
        run_dir1 = Path("/tmp/test_run_1")
        run_dir2 = Path("/tmp/test_run_2")

        sample_job_data["run_dir"] = run_dir1
        temp_registry.register_job(**sample_job_data)

        sample_job_data["run_dir"] = run_dir2
        temp_registry.register_job(**sample_job_data)

        history = temp_registry.get_job_history(run_dir=run_dir1)
        assert len(history) == 1
        assert Path(history[0].run_dir).resolve() == run_dir1.resolve()

    def test_get_job_history_with_limit(self, temp_registry, sample_job_data):
        """Test retrieving job history with limit."""
        for i in range(5):
            sample_job_data["run_dir"] = Path(f"/tmp/test_run_{i}")
            temp_registry.register_job(**sample_job_data)

        history = temp_registry.get_job_history(limit=3)
        assert len(history) == 3

    def test_archive_completed_jobs(self, temp_registry, sample_job_data):
        """Test archiving old completed jobs."""
        job_id = temp_registry.register_job(**sample_job_data)

        # Complete the job with an old completion date
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        temp_registry.update_job(job_id, status="completed", completed_at=old_date)

        # Archive jobs older than 30 days
        cutoff = datetime.now() - timedelta(days=30)
        count = temp_registry.archive_completed_jobs(before=cutoff)

        assert count == 1

        # Verify job is archived
        job = temp_registry.get_job_by_id(job_id)
        assert job.status == "archived"

    def test_corrupted_jsonl_lines(self, temp_registry, sample_job_data):
        """Test handling of corrupted JSONL lines."""
        job_id = temp_registry.register_job(**sample_job_data)

        # Add corrupted line
        with open(temp_registry.registry_path, "a") as f:
            f.write("this is not valid json\n")

        # Should still be able to read valid entries
        jobs = temp_registry.get_active_jobs()
        assert len(jobs) == 1
        assert jobs[0].job_id == job_id

    def test_get_job_by_id(self, temp_registry, sample_job_data):
        """Test retrieving job by ID."""
        job_id = temp_registry.register_job(**sample_job_data)
        job = temp_registry.get_job_by_id(job_id)

        assert job is not None
        assert job.job_id == job_id
        assert job.pid == sample_job_data["pid"]

    def test_get_job_by_id_returns_latest(self, temp_registry, sample_job_data):
        """Test that get_job_by_id returns latest record after updates."""
        job_id = temp_registry.register_job(**sample_job_data)
        temp_registry.update_job(job_id, status="completed", exit_code=0)

        job = temp_registry.get_job_by_id(job_id)
        assert job.status == "completed"
        assert job.exit_code == 0

    def test_concurrent_writes(self, tmp_path, sample_job_data):
        """Test that concurrent writes don't corrupt the registry."""

        def write_job(registry_path, pid):
            registry = JobRegistry(registry_path)
            data = sample_job_data.copy()
            data["pid"] = pid
            data["run_dir"] = Path(f"/tmp/test_{pid}")
            registry.register_job(**data)

        registry_path = tmp_path / "concurrent_test.jsonl"

        # Spawn multiple processes writing simultaneously
        processes = []
        for i in range(5):
            p = multiprocessing.Process(target=write_job, args=(registry_path, 1000 + i))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Verify all jobs were written
        registry = JobRegistry(registry_path)
        history = registry.get_job_history()
        assert len(history) == 5

        # Verify no duplicate job IDs
        job_ids = [job.job_id for job in history]
        assert len(job_ids) == len(set(job_ids))


@pytest.mark.unit
class TestJobRegistryEdgeCases:
    """Test edge cases for JobRegistry."""

    def test_empty_registry(self, temp_registry):
        """Test operations on empty registry."""
        assert len(temp_registry.get_active_jobs()) == 0
        assert len(temp_registry.get_job_history()) == 0
        assert temp_registry.get_job_by_id("nonexistent") is None

    def test_update_nonexistent_job(self, temp_registry):
        """Test updating a job that doesn't exist."""
        # Should not raise an error, just log a warning
        temp_registry.update_job("nonexistent-id", status="completed")

    def test_path_normalization(self, temp_registry, sample_job_data):
        """Test that paths are normalized correctly."""
        sample_job_data["run_dir"] = Path("/tmp/../tmp/test")
        job_id = temp_registry.register_job(**sample_job_data)

        # Should find by normalized path
        job = temp_registry.get_job_by_run_dir(Path("/tmp/test"))
        assert job.job_id == job_id
