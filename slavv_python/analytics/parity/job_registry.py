"""Persistent storage and query interface for parity job metadata."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import fasteners
except ImportError:
    fasteners = None

logger = logging.getLogger(__name__)


@dataclass
class ParityJobRecord:
    """Metadata record for a parity experiment job."""

    job_id: str
    pid: int
    run_dir: str
    oracle_root: str
    stage: str
    command: str
    started_at: str  # ISO format datetime string
    last_seen_at: str  # ISO format datetime string
    completed_at: str | None = None
    exit_code: int | None = None
    status: str = "running"  # 'running', 'completed', 'failed', 'killed'
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParityJobRecord:
        """Create record from dictionary."""
        return cls(**data)


class JobRegistry:
    """Persistent JSONL-based storage for parity job records."""

    def __init__(self, registry_path: Path | None = None):
        """
        Initialize job registry.

        Args:
            registry_path: Path to JSONL registry file. Defaults to
                workspace/scratch/job_registry.jsonl
        """
        if registry_path is None:
            workspace = Path("workspace/scratch")
            workspace.mkdir(parents=True, exist_ok=True)
            registry_path = workspace / "job_registry.jsonl"

        self.registry_path = Path(registry_path)
        self.lock_path = self.registry_path.with_suffix(".lock")

        # Ensure registry file exists
        if not self.registry_path.exists():
            self.registry_path.touch()

        logger.debug(f"JobRegistry initialized at {self.registry_path}")

    def _acquire_lock(self):
        """Acquire file lock for thread-safe operations."""
        if fasteners is None:
            logger.warning("fasteners not available, skipping file locking")
            return None
        return fasteners.InterProcessLock(str(self.lock_path))

    def _read_all_records(self) -> list[ParityJobRecord]:
        """Read all records from registry file."""
        records = []
        with open(self.registry_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(ParityJobRecord.from_dict(data))
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    logger.warning(f"Skipping corrupted line {line_num} in registry: {e}")
                    continue
        return records

    def _append_record(self, record: ParityJobRecord) -> None:
        """Append a record to the registry file."""
        with open(self.registry_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def register_job(
        self,
        pid: int,
        run_dir: Path,
        oracle_root: Path,
        stage: str,
        command: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Register a new parity job.

        Args:
            pid: Process ID of the job
            run_dir: Run directory path
            oracle_root: Oracle root path
            stage: Stage name (energy, vertices, edges, network, sequence)
            command: Full CLI command
            metadata: Additional job metadata

        Returns:
            Job ID (UUID string)
        """
        lock = self._acquire_lock()
        try:
            if lock:
                lock.acquire()

            job_id = str(uuid.uuid4())
            now = datetime.now().isoformat()

            record = ParityJobRecord(
                job_id=job_id,
                pid=pid,
                run_dir=str(run_dir),
                oracle_root=str(oracle_root),
                stage=stage,
                command=command,
                started_at=now,
                last_seen_at=now,
                status="running",
                metadata=metadata or {},
            )

            self._append_record(record)
            logger.info(f"Registered job {job_id} (PID {pid}, stage {stage})")
            return job_id

        finally:
            if lock:
                lock.release()

    def update_job(self, job_id: str, **updates) -> None:
        """
        Update a job record.

        Args:
            job_id: Job ID to update
            **updates: Fields to update (status, completed_at, exit_code, etc.)
        """
        lock = self._acquire_lock()
        try:
            if lock:
                lock.acquire()

            # Find the latest record for this job
            records = self._read_all_records()
            target_record = None
            for record in reversed(records):
                if record.job_id == job_id:
                    target_record = record
                    break

            if target_record is None:
                logger.warning(f"Job {job_id} not found in registry")
                return

            # Always update last_seen_at unless explicitly provided
            if "last_seen_at" not in updates:
                updates["last_seen_at"] = datetime.now().isoformat()

            # Create updated record
            record_dict = target_record.to_dict()
            record_dict.update(updates)
            updated_record = ParityJobRecord.from_dict(record_dict)

            # Append updated record
            self._append_record(updated_record)
            logger.debug(f"Updated job {job_id}: {updates}")

        finally:
            if lock:
                lock.release()

    def get_active_jobs(self) -> list[ParityJobRecord]:
        """
        Get all active (running) jobs.

        Returns:
            List of job records with status='running'
        """
        lock = self._acquire_lock()
        try:
            if lock:
                lock.acquire()

            records = self._read_all_records()

            # Build map of job_id -> latest record
            job_map: dict[str, ParityJobRecord] = {}
            for record in records:
                if (
                    record.job_id not in job_map
                    or record.last_seen_at >= job_map[record.job_id].last_seen_at
                ):
                    job_map[record.job_id] = record

            # Filter for running jobs
            active = [r for r in job_map.values() if r.status == "running"]
            return sorted(active, key=lambda r: r.started_at)

        finally:
            if lock:
                lock.release()

    def get_job_by_run_dir(self, run_dir: Path) -> ParityJobRecord | None:
        """
        Get the latest job for a specific run directory.

        Args:
            run_dir: Run directory path

        Returns:
            Latest job record for that directory, or None if not found
        """
        lock = self._acquire_lock()
        try:
            if lock:
                lock.acquire()

            records = self._read_all_records()
            run_dir_str = str(Path(run_dir).resolve())

            # Find latest record for this run_dir
            matching_records = [
                r for r in records if Path(r.run_dir).resolve() == Path(run_dir_str)
            ]

            if not matching_records:
                return None

            # Build map of job_id -> latest record
            job_map: dict[str, ParityJobRecord] = {}
            for record in matching_records:
                if (
                    record.job_id not in job_map
                    or record.last_seen_at > job_map[record.job_id].last_seen_at
                ):
                    job_map[record.job_id] = record

            # Return most recent active job, or most recent completed job
            active = [r for r in job_map.values() if r.status == "running"]
            if active:
                return max(active, key=lambda r: r.started_at)

            all_jobs = list(job_map.values())
            return max(all_jobs, key=lambda r: r.started_at)

        finally:
            if lock:
                lock.release()

    def get_job_history(
        self, run_dir: Path | None = None, limit: int | None = None
    ) -> list[ParityJobRecord]:
        """
        Get job history, optionally filtered by run directory.

        Args:
            run_dir: Optional run directory filter
            limit: Maximum number of jobs to return (most recent first)

        Returns:
            List of job records (latest state for each job)
        """
        lock = self._acquire_lock()
        try:
            if lock:
                lock.acquire()

            records = self._read_all_records()

            # Filter by run_dir if specified
            if run_dir is not None:
                run_dir_str = str(Path(run_dir).resolve())
                records = [r for r in records if Path(r.run_dir).resolve() == Path(run_dir_str)]

            # Build map of job_id -> latest record
            job_map: dict[str, ParityJobRecord] = {}
            for record in records:
                if (
                    record.job_id not in job_map
                    or record.last_seen_at > job_map[record.job_id].last_seen_at
                ):
                    job_map[record.job_id] = record

            # Sort by started_at (most recent first)
            jobs = sorted(job_map.values(), key=lambda r: r.started_at, reverse=True)

            if limit is not None:
                jobs = jobs[:limit]

            return jobs

        finally:
            if lock:
                lock.release()

    def archive_completed_jobs(self, before: datetime) -> int:
        """
        Mark old completed jobs as archived.

        Args:
            before: Archive jobs completed before this datetime

        Returns:
            Number of jobs archived
        """
        lock = self._acquire_lock()
        try:
            if lock:
                lock.acquire()

            records = self._read_all_records()
            archived_count = 0

            # Build map of job_id -> latest record
            job_map: dict[str, ParityJobRecord] = {}
            for record in records:
                if (
                    record.job_id not in job_map
                    or record.last_seen_at > job_map[record.job_id].last_seen_at
                ):
                    job_map[record.job_id] = record

            before_iso = before.isoformat()
            for job_id, record in job_map.items():
                if (
                    record.status in ("completed", "failed")
                    and record.completed_at
                    and record.completed_at < before_iso
                ):
                    # Update status to archived
                    self.update_job(job_id, status="archived")
                    archived_count += 1

            logger.info(f"Archived {archived_count} jobs completed before {before}")
            return archived_count

        finally:
            if lock:
                lock.release()

    def get_job_by_id(self, job_id: str) -> ParityJobRecord | None:
        """
        Get latest record for a specific job ID.

        Args:
            job_id: Job ID to look up

        Returns:
            Latest job record, or None if not found
        """
        lock = self._acquire_lock()
        try:
            if lock:
                lock.acquire()

            records = self._read_all_records()
            matching = [r for r in records if r.job_id == job_id]

            if not matching:
                return None

            # Return latest record
            return max(matching, key=lambda r: r.last_seen_at)

        finally:
            if lock:
                lock.release()
