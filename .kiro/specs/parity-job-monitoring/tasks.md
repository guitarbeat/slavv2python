# Tasks: Automated Parity Job Monitoring System

## Phase 1: Core Infrastructure

### Task 1: Implement JobRegistry
**File**: `slavv_python/analytics/parity/job_registry.py`
**Dependencies**: None
**Estimated Effort**: 3 hours

- [ ] Define `ParityJobRecord` dataclass with all required fields
- [ ] Implement JSONL file storage backend at `workspace/scratch/job_registry.jsonl`
- [ ] Add file locking using `fasteners.InterProcessLock`
- [ ] Implement `register_job()` - append new record with UUID
- [ ] Implement `update_job()` - find by job_id, append updated record
- [ ] Implement `get_active_jobs()` - filter records by status='running'
- [ ] Implement `get_job_by_run_dir()` - find latest record for run directory
- [ ] Implement `get_job_history()` - query all records, optionally filter by run_dir
- [ ] Implement `archive_completed_jobs()` - mark old completed jobs as archived
- [ ] Add logging for all operations
- [ ] Handle corrupted JSONL lines gracefully (skip and log warning)

**Acceptance Criteria**:
- Registry survives concurrent writes from multiple processes
- Queries return correct active/historical jobs
- File format is human-readable JSONL

---

### Task 2: Implement MonitorDaemon
**File**: `slavv_python/analytics/parity/monitor_daemon.py`
**Dependencies**: Task 1
**Estimated Effort**: 4 hours

- [ ] Create `MonitorDaemon` class with start/stop/status methods
- [ ] Implement detached process spawning (subprocess with DETACHED_PROCESS on Windows)
- [ ] Write PID to `workspace/scratch/monitor_daemon.pid`
- [ ] Configure logging to `workspace/scratch/monitor_daemon.log` with rotation
- [ ] Implement main polling loop (30-second interval, configurable)
- [ ] For each active job: check if PID exists and process is alive
- [ ] Update `last_seen_at` timestamp for running jobs
- [ ] Mark completed jobs and trigger notifications
- [ ] Implement desktop notification using `win10toast`
- [ ] Add fallback logging if `win10toast` unavailable
- [ ] Implement auto-termination if no active jobs for 1 hour
- [ ] Add signal handlers for graceful shutdown (SIGTERM, SIGINT)
- [ ] Write heartbeat to `workspace/scratch/monitor_daemon_heartbeat.json`

**Acceptance Criteria**:
- Daemon starts as detached process
- Survives parent process termination
- Sends desktop notifications on Windows
- Logs all events to file
- Self-terminates when idle

---

### Task 3: Add Helper Functions for Process Management
**File**: `slavv_python/analytics/parity/process_utils.py`
**Dependencies**: None
**Estimated Effort**: 1 hour

- [ ] Implement `is_process_alive(pid: int) -> bool` using `psutil`
- [ ] Implement `get_process_info(pid: int) -> Optional[Dict]` - name, cmdline, start time
- [ ] Implement `kill_process_tree(pid: int)` - terminate process and children
- [ ] Implement `ensure_monitor_daemon_running()` - start daemon if not running
- [ ] Add validation: check process name contains 'python' to prevent PID reuse false positives

**Acceptance Criteria**:
- Functions handle non-existent PIDs gracefully
- Process tree termination works recursively
- PID validation prevents false positives from PID reuse

---

### Task 4: Integrate with parity_experiment.py
**File**: `scripts/cli/parity_experiment.py`
**Dependencies**: Tasks 1, 2, 3
**Estimated Effort**: 3 hours

- [ ] Add `--monitor` flag to `resume-exact-run` command
- [ ] Add `--force-kill` flag to terminate active writers
- [ ] Import `JobRegistry` and helper functions
- [ ] Add duplicate writer check before starting detached subprocess
- [ ] If duplicate detected and `--force-kill` not set, raise error with helpful message
- [ ] If `--force-kill` set, terminate active job and proceed
- [ ] After spawning detached subprocess, register job in registry if `--monitor` enabled
- [ ] Call `ensure_monitor_daemon_running()` after registering job
- [ ] Print job ID and monitoring confirmation message
- [ ] Update help text and docstrings

**Acceptance Criteria**:
- `--monitor` flag registers job correctly
- Duplicate writer detection prevents concurrent writes
- `--force-kill` terminates existing job before starting new one
- Daemon starts automatically when first job is monitored

---

### Task 5: Implement `slavv jobs` CLI Command
**File**: `slavv_python/interface/cli/jobs.py`
**Dependencies**: Tasks 1, 2, 3
**Estimated Effort**: 3 hours

- [ ] Create new CLI entry point: `slavv jobs`
- [ ] Implement `slavv jobs list` subcommand
  - [ ] Query `registry.get_active_jobs()`
  - [ ] Format as table using `tabulate`
  - [ ] Show: Job ID (short), PID, Run Dir (truncated), Stage, Status, Started, Duration
  - [ ] Handle empty results gracefully
- [ ] Implement `slavv jobs history` subcommand
  - [ ] Accept optional `--run-dir PATH` filter
  - [ ] Query `registry.get_job_history()`
  - [ ] Format as table with completed jobs
  - [ ] Show: Job ID, PID, Started, Stage, Exit Code, Duration, Status
- [ ] Implement `slavv jobs kill <job-id>` subcommand
  - [ ] Look up job by ID
  - [ ] Call `kill_process_tree(job.pid)`
  - [ ] Update job status to 'killed'
  - [ ] Print confirmation
- [ ] Implement `slavv jobs daemon status` subcommand
  - [ ] Read daemon PID file
  - [ ] Check if process alive
  - [ ] Read heartbeat file for last update time
  - [ ] Print: PID, Status (running/stopped), Uptime, Active Jobs Count
- [ ] Implement `slavv jobs daemon restart` subcommand
  - [ ] Stop existing daemon if running
  - [ ] Start new daemon instance
  - [ ] Print confirmation
- [ ] Register all subcommands in `slavv_python/interface/cli/main.py`
- [ ] Add help text and examples

**Acceptance Criteria**:
- All subcommands work correctly
- Output is well-formatted and readable
- Error messages are clear and actionable
- Help text includes examples

---

### Task 6: Update Entry Point Registration
**File**: `pyproject.toml`, `slavv_python/interface/cli/main.py`
**Dependencies**: Task 5
**Estimated Effort**: 30 minutes

- [ ] Add `jobs = slavv_python.interface.cli.jobs:main` to console_scripts in pyproject.toml
- [ ] Update main CLI dispatcher to include `jobs` command
- [ ] Ensure `pip install -e .` picks up new command

**Acceptance Criteria**:
- `slavv jobs` command is available after `pip install -e .`

---

## Phase 2: Testing

### Task 7: Unit Tests for JobRegistry
**File**: `tests/unit/analytics/parity/test_job_registry.py`
**Dependencies**: Task 1
**Estimated Effort**: 2 hours

- [ ] Test job registration creates correct record
- [ ] Test job updates append correctly
- [ ] Test active jobs query returns only running jobs
- [ ] Test run-dir query returns latest job for that directory
- [ ] Test history query returns all jobs, filtered correctly
- [ ] Test archive functionality marks old jobs
- [ ] Test concurrent writes don't corrupt file (multi-process test)
- [ ] Test corrupted JSONL lines are handled gracefully
- [ ] Test file locking prevents race conditions

**Acceptance Criteria**:
- All tests pass
- Code coverage > 90% for job_registry.py

---

### Task 8: Unit Tests for MonitorDaemon
**File**: `tests/unit/analytics/parity/test_monitor_daemon.py`
**Dependencies**: Task 2
**Estimated Effort**: 2 hours

- [ ] Test daemon starts and writes PID file
- [ ] Test daemon detects completed jobs
- [ ] Test daemon updates last_seen_at for running jobs
- [ ] Test daemon terminates when idle (mock time.sleep)
- [ ] Test notification dispatch (mock win10toast)
- [ ] Test heartbeat updates
- [ ] Test graceful shutdown on SIGTERM

**Acceptance Criteria**:
- All tests pass
- Tests use mocks to avoid spawning real daemons
- Code coverage > 85% for monitor_daemon.py

---

### Task 9: Integration Tests
**File**: `tests/integration/parity/test_monitored_parity_run.py`
**Dependencies**: Tasks 1-6
**Estimated Effort**: 3 hours

- [ ] Test end-to-end monitored job lifecycle
  - [ ] Start synthetic parity run with `--monitor`
  - [ ] Verify job registered in registry
  - [ ] Verify daemon is running
  - [ ] Wait for job completion
  - [ ] Verify notification was attempted (check logs)
  - [ ] Verify job marked completed in registry
- [ ] Test duplicate writer prevention
  - [ ] Start monitored job
  - [ ] Attempt to start another job on same run-dir
  - [ ] Verify second attempt fails with clear error
- [ ] Test `--force-kill` behavior
  - [ ] Start monitored job
  - [ ] Start second job with `--force-kill`
  - [ ] Verify first job terminated
  - [ ] Verify second job proceeds
- [ ] Test daemon recovery after crash
  - [ ] Start monitored job
  - [ ] Kill daemon process
  - [ ] Start another monitored job
  - [ ] Verify daemon restarts and picks up both jobs

**Acceptance Criteria**:
- All integration tests pass
- Tests use synthetic small volumes for speed (< 1 minute per test)
- Tests clean up temporary files and processes

---

### Task 10: Manual Testing on Windows
**Estimated Effort**: 1 hour

- [ ] Install `win10toast` on Windows machine
- [ ] Start long-running parity job with `--monitor`
- [ ] Verify desktop notification appears at start
- [ ] Leave terminal closed for duration
- [ ] Verify desktop notification appears at completion
- [ ] Test `slavv jobs list` shows running job
- [ ] Test `slavv jobs history` shows completed job
- [ ] Test `slavv jobs kill` terminates job
- [ ] Test daemon persists across terminal restarts

**Acceptance Criteria**:
- All manual test scenarios pass on Windows 10/11

---

## Phase 3: Documentation

### Task 11: Update Documentation
**Files**: Various
**Dependencies**: Tasks 1-10
**Estimated Effort**: 2 hours

- [ ] Update `docs/reference/workflow/PARITY_PRE_GATE.md`
  - [ ] Add `--monitor` flag to example commands
  - [ ] Document `slavv jobs` commands
- [ ] Update `docs/reference/workflow/PARITY_CERTIFICATION_GUIDE.md`
  - [ ] Add monitoring best practices
  - [ ] Document job history queries
- [ ] Update `docs/reference/core/EXACT_PROOF_FINDINGS.md`
  - [ ] Update "Cold-start protocol" section
  - [ ] Reference `slavv jobs` for status checks
- [ ] Create new doc: `docs/reference/workflow/PARITY_JOB_MONITORING.md`
  - [ ] Architecture overview
  - [ ] Usage examples
  - [ ] Troubleshooting guide
- [ ] Update `README.md` with monitoring feature
- [ ] Add monitoring to `scripts/cli/README.md` (if exists)

**Acceptance Criteria**:
- All references to PID checking updated to use `slavv jobs`
- New monitoring doc is complete and clear
- Examples include `--monitor` flag

---

### Task 12: Update CHANGELOG
**File**: `docs/CHANGELOG.md`
**Dependencies**: Task 11
**Estimated Effort**: 15 minutes

- [ ] Add entry for monitoring feature under "Added" section
- [ ] Document new `slavv jobs` commands
- [ ] Document new `--monitor` and `--force-kill` flags

**Acceptance Criteria**:
- CHANGELOG entry is clear and includes examples

---

## Phase 4: Deployment

### Task 13: Add Dependencies to pyproject.toml
**File**: `pyproject.toml`
**Dependencies**: None
**Estimated Effort**: 15 minutes

- [ ] Add `win10toast>=0.9` to dependencies (optional for Windows)
- [ ] Add `fasteners` if not already present
- [ ] Add `psutil` if not already present
- [ ] Add `tabulate` if not already present
- [ ] Update installation instructions in README if needed

**Acceptance Criteria**:
- Dependencies install correctly with `pip install -e .`

---

### Task 14: Create Migration Script (if needed)
**File**: `scripts/maintenance/migrate_legacy_pid_files.py`
**Dependencies**: Task 1
**Estimated Effort**: 1 hour

- [ ] Scan `workspace/scratch/` for `*.pid` files
- [ ] Scan `workspace/runs/*/99_Metadata/parity_job.json` files
- [ ] Import discovered jobs into new registry
- [ ] Mark all as historical (not actively monitored)
- [ ] Print summary of migrated jobs

**Acceptance Criteria**:
- Script runs without errors
- Historical jobs appear in `slavv jobs history`

---

## Summary

**Total Estimated Effort**: ~25 hours

**Critical Path**:
1. JobRegistry (Task 1) → 3h
2. MonitorDaemon (Task 2) → 4h
3. Process Utils (Task 3) → 1h
4. CLI Integration (Task 4) → 3h
5. Jobs Command (Task 5) → 3h
6. Testing (Tasks 7-10) → 8h
7. Documentation (Tasks 11-12) → 2.25h
8. Deployment (Tasks 13-14) → 1.25h

**Milestones**:
- **M1**: Core implementation complete (Tasks 1-6) → ~14 hours
- **M2**: Testing complete (Tasks 7-10) → +8 hours
- **M3**: Documentation complete (Tasks 11-12) → +2.25 hours
- **M4**: Ready for production (Tasks 13-14) → +1.25 hours

**Risk Factors**:
- Windows notification library compatibility issues
- Daemon process stability on different Windows versions
- JSONL file corruption under high concurrency
- PID reuse edge cases

**Mitigation**:
- Extensive testing on Windows 10/11
- File locking and atomic writes
- Process name validation to prevent PID reuse false positives
- Graceful degradation if notifications unavailable
