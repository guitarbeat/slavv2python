# Implementation Status: Parity Job Monitoring System

**Started**: 2026-06-09  
**Current Status**: Phase 1 Core Infrastructure - Partially Complete

---

## ✅ Completed Tasks

### Task 1: JobRegistry Implementation
**File**: `slavv_python/analytics/parity/job_registry.py`  
**Status**: ✅ COMPLETE

- [x] `ParityJobRecord` dataclass with all fields
- [x] JSONL file storage backend
- [x] File locking with `fasteners.InterProcessLock`
- [x] `register_job()` - append new record with UUID
- [x] `update_job()` - find by job_id, append updated record
- [x] `get_active_jobs()` - filter by status='running'
- [x] `get_job_by_run_dir()` - find latest for run directory
- [x] `get_job_history()` - query all records
- [x] `archive_completed_jobs()` - mark old jobs
- [x] Error handling for corrupted JSONL lines

### Task 2: MonitorDaemon Implementation  
**File**: `slavv_python/analytics/parity/monitor_daemon.py`  
**Status**: ✅ COMPLETE

- [x] `MonitorDaemon` class with start/stop/status methods
- [x] Detached process spawning (Windows DETACHED_PROCESS)
- [x] PID file management
- [x] Logging to file with rotation capability
- [x] Main polling loop (30-second interval)
- [x] Job liveness checking
- [x] Desktop notifications with `win10toast`
- [x] Fallback logging if notifications unavailable
- [x] Auto-termination when idle
- [x] Signal handlers for graceful shutdown
- [x] Heartbeat file

### Task 3: Process Utilities
**File**: `slavv_python/analytics/parity/process_utils.py`  
**Status**: ✅ COMPLETE

- [x] `is_process_alive()` using psutil
- [x] `get_process_info()` - process metadata
- [x] `kill_process_tree()` - recursive termination
- [x] `ensure_monitor_daemon_running()` - auto-start daemon
- [x] `is_python_process()` - PID reuse validation
- [x] Helper functions for daemon PID management

### Task 13: Dependencies (Partial)
**File**: `pyproject.toml`  
**Status**: ✅ COMPLETE

- [x] Added `fasteners>=0.18.0` to core dependencies
- [x] Added `tabulate>=0.9.0` to core dependencies  
- [x] Added `win10toast>=0.9` to workspace extras (Windows only)
- [x] `psutil` already present

---

## 🚧 Remaining Tasks

### Phase 1: Core Infrastructure

#### Task 4: Integrate with parity_experiment.py [CRITICAL]
**File**: `scripts/cli/parity_experiment.py`  
**Effort**: 3 hours  
**Priority**: HIGH

**Steps**:
1. Add `--monitor` and `--force-kill` flags to argparse
2. Import JobRegistry and process_utils
3. Add duplicate writer check before subprocess spawn
4. Register job if `--monitor` enabled
5. Call `ensure_monitor_daemon_running()`
6. Print job ID confirmation

**Code Template**:
```python
# In resume_exact_run() before spawning subprocess:
if monitor:
    registry = JobRegistry()
    active_job = registry.get_job_by_run_dir(dest_run_root)
    if active_job and is_process_alive(active_job.pid):
        if not force_kill:
            raise RuntimeError(f"Active writer (PID {active_job.pid}). Use --force-kill")
        kill_process_tree(active_job.pid)
    
    # After spawning proc:
    job_id = registry.register_job(
        pid=proc.pid,
        run_dir=dest_run_root,
        oracle_root=oracle_root,
        stage=infer_stage(...),
        command=' '.join(sys.argv),
    )
    ensure_monitor_daemon_running()
    print(f"Job registered for monitoring (ID: {job_id})")
```

#### Task 5: Implement `slavv jobs` CLI Command [CRITICAL]
**File**: `slavv_python/interface/cli/jobs.py`  
**Effort**: 3 hours  
**Priority**: HIGH

**Steps**:
1. Create new file with argparse subcommands
2. Implement `list` - query active jobs, format with tabulate
3. Implement `history` - query history, accept --run-dir filter
4. Implement `kill <job-id>` - terminate job
5. Implement `daemon status` - show daemon PID/uptime
6. Implement `daemon restart` - stop and start daemon

**Code Template**:
```python
def cmd_list(args):
    registry = JobRegistry()
    jobs = registry.get_active_jobs()
    if not jobs:
        print("No active jobs")
        return
    
    rows = []
    for job in jobs:
        duration = datetime.now() - datetime.fromisoformat(job.started_at)
        rows.append([
            job.job_id[:8],
            job.pid,
            Path(job.run_dir).name,
            job.stage,
            job.status,
            job.started_at,
            format_duration(duration),
        ])
    
    print(tabulate(rows, headers=["Job ID", "PID", "Run Dir", "Stage", "Status", "Started", "Duration"]))
```

#### Task 6: Entry Point Registration
**File**: `pyproject.toml`, `slavv_python/interface/cli/main.py`  
**Effort**: 30 minutes

Add `jobs` subcommand to main CLI dispatcher.

---

### Phase 2: Testing

#### Task 7: Unit Tests for JobRegistry
**File**: `tests/unit/analytics/parity/test_job_registry.py`  
**Effort**: 2 hours

Test all CRUD operations, locking, history queries.

#### Task 8: Unit Tests for MonitorDaemon  
**File**: `tests/unit/analytics/parity/test_monitor_daemon.py`  
**Effort**: 2 hours

Test daemon lifecycle, job detection, notifications (mocked).

#### Task 9: Integration Tests
**File**: `tests/integration/parity/test_monitored_parity_run.py`  
**Effort**: 3 hours

End-to-end test with synthetic volume, duplicate writer prevention.

#### Task 10: Manual Testing on Windows
**Effort**: 1 hour

Test desktop notifications, cross-terminal persistence.

---

### Phase 3: Documentation

#### Task 11: Update Documentation
**Files**: Various docs  
**Effort**: 2 hours

Update parity guides with `--monitor` flag, create monitoring reference doc.

#### Task 12: Update CHANGELOG
**File**: `docs/CHANGELOG.md`  
**Effort**: 15 minutes

Document new feature.

---

### Phase 4: Deployment

#### Task 14: Migration Script (Optional)
**File**: `scripts/maintenance/migrate_legacy_pid_files.py`  
**Effort**: 1 hour

Import historical jobs from existing PID files.

---

## 🎯 Critical Path to Minimal Viable Product

To get a working system ASAP, focus on:

1. **Task 4**: CLI Integration (`parity_experiment.py --monitor`) - 3h
2. **Task 5**: `slavv jobs` commands - 3h
3. **Task 6**: Entry point registration - 30m
4. **Install dependencies**: `pip install -e ".[workspace]"` - 5m
5. **Manual smoke test**: Start monitored job, check `slavv jobs list` - 15m

**Total to MVP**: ~7 hours

After MVP works, add tests (Tasks 7-10) and documentation (Tasks 11-12).

---

## 🐛 Known Issues & TODOs

1. **Daemon restart on failure**: Currently no automatic restart if daemon crashes
2. **Log rotation**: `monitor_daemon.log` will grow unbounded
3. **Registry cleanup**: JSONL will grow; need archival/rotation
4. **Cross-platform**: Notifications only work on Windows (need Linux/macOS fallback)
5. **Error recovery**: If registry gets corrupted, no automatic repair

---

## 📋 Quick Start (After Tasks 4-6 Complete)

```powershell
# Install with monitoring dependencies
pip install -e ".[workspace]"

# Start a monitored parity job
python scripts/cli/parity_experiment.py resume-exact-run \
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact \
  --oracle-root workspace/oracles/180709_E_crop_M \
  --force-rerun-from energy \
  --stop-after network \
  --monitor

# Check active jobs
slavv jobs list

# Check job history
slavv jobs history --run-dir workspace/runs/oracle_180709_E/crop_M_exact

# Kill a job
slavv jobs kill <job-id>

# Check daemon status
slavv jobs daemon status
```

---

## 🚀 Next Steps

**Immediate**: Complete Tasks 4-6 to reach MVP (~7 hours work)

**Short-term**: Add tests (Tasks 7-10) for reliability (~8 hours)

**Long-term**: 
- Add Slack/email notifications (future enhancement)
- Integrate with Streamlit dashboard
- Add resource usage tracking (CPU, memory, disk I/O)
- Cross-platform notification support

---

**Last Updated**: 2026-06-09  
**Implementation by**: Kiro AI Agent
