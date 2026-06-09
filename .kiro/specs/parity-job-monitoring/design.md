# Design: Automated Parity Job Monitoring System

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User starts job with --monitor            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  parity_experiment.py                                        │
│  - Validates no active writer on run-dir                     │
│  - Spawns detached Python subprocess                         │
│  - Registers job in JobRegistry                              │
│  - Starts MonitorDaemon if not running                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌─────────────────────┐    ┌──────────────────────────┐
│  Detached Job        │    │  MonitorDaemon           │
│  (subprocess)        │    │  (background process)    │
│  - Runs parity exp   │    │  - Polls job registry    │
│  - Writes to         │    │  - Checks PID liveness   │
│    99_Metadata/      │    │  - Sends notifications   │
│  - Updates progress  │    │  - Archives completed    │
└──────────────────────┘    └──────────────────────────┘
                                      │
                  ┌───────────────────┴───────────────────┐
                  │                                       │
                  ▼                                       ▼
         ┌─────────────────┐                  ┌─────────────────┐
         │  JobRegistry     │                  │  Notifications  │
         │  (JSONL file)    │                  │  (win10toast)   │
         │  - Active jobs   │                  │  - Toast popups │
         │  - Job history   │                  │  - Tray icon    │
         └─────────────────┘                  └─────────────────┘
```

## Component Design

### 1. JobRegistry

**Location**: `slavv_python/analytics/parity/job_registry.py`

**Purpose**: Persistent storage and query interface for parity job metadata.

**Data Model**:
```python
@dataclass
class ParityJobRecord:
    job_id: str                    # UUID
    pid: int
    run_dir: Path
    oracle_root: Path
    stage: str                     # 'energy', 'vertices', 'edges', 'network', 'sequence'
    command: str                   # Full CLI command
    started_at: datetime
    last_seen_at: datetime
    completed_at: Optional[datetime]
    exit_code: Optional[int]
    status: str                    # 'running', 'completed', 'failed', 'killed'
    metadata: Dict[str, Any]       # Extra context (params, flags, etc.)
```

**Storage**: `workspace/scratch/job_registry.jsonl` (append-only, one JSON object per line)

**Operations**:
- `register_job(record: ParityJobRecord) -> str` - Add new job
- `update_job(job_id: str, **updates)` - Update fields
- `get_active_jobs() -> List[ParityJobRecord]` - List running jobs
- `get_job_by_run_dir(run_dir: Path) -> Optional[ParityJobRecord]` - Check for active writer
- `get_job_history(run_dir: Optional[Path] = None) -> List[ParityJobRecord]` - Query history
- `archive_completed_jobs(before: datetime)` - Cleanup old entries

**Thread Safety**: File-based locking via `fasteners.InterProcessLock`

### 2. MonitorDaemon

**Location**: `slavv_python/analytics/parity/monitor_daemon.py`

**Purpose**: Background process that polls job registry and manages notifications.

**Lifecycle**:
- Started automatically when first monitored job is registered
- Persists across terminal/IDE restarts (detached process)
- Writes PID to `workspace/scratch/monitor_daemon.pid`
- Logs to `workspace/scratch/monitor_daemon.log`

**Polling Loop** (every 30 seconds):
```
FOR EACH active_job IN registry:
  IF PID exists and process running:
    UPDATE last_seen_at = now
    CHECK run-local 99_Metadata/parity_job.json for progress
  ELSE:
    MARK job as completed/failed
    SEND desktop notification (success/failure)
    ARCHIVE job record
```

**Notification Strategy**:
- Use `win10toast` library (Windows 10/11)
- Fallback to log-only if toast unavailable (CI, WSL, etc.)
- Notification content: Job ID, stage, duration, success/failure

**Health Check**:
- Self-terminates if no active jobs for 1 hour (conserve resources)
- Restart automatically when new monitored job starts

### 3. CLI Integration

**Location**: `scripts/cli/parity_experiment.py`

**New Flag**: `--monitor` (optional, default: False)

**Modified Workflow**:
```python
def resume_exact_run(..., monitor: bool = False):
    # 1. Existing preflight checks
    
    # 2. NEW: Check for active writer
    registry = JobRegistry()
    active_job = registry.get_job_by_run_dir(dest_run_root)
    if active_job and is_process_alive(active_job.pid):
        raise RuntimeError(
            f"Run directory has active writer (PID {active_job.pid}).\n"
            f"Job started: {active_job.started_at}\n"
            f"Use --force-kill to terminate, or wait for completion."
        )
    
    # 3. Start detached subprocess
    cmd = build_resume_exact_command(...)
    proc = subprocess.Popen(cmd, ...)
    
    # 4. NEW: Register job if --monitor
    if monitor:
        job_record = ParityJobRecord(
            job_id=str(uuid.uuid4()),
            pid=proc.pid,
            run_dir=dest_run_root,
            oracle_root=oracle_root,
            stage=infer_stage_from_flags(...),
            command=' '.join(sys.argv),
            started_at=datetime.now(),
            last_seen_at=datetime.now(),
            status='running',
            metadata={...}
        )
        registry.register_job(job_record)
        ensure_monitor_daemon_running()
        print(f"Job registered for monitoring (ID: {job_record.job_id})")
    
    # 5. Return PID as before
    return proc.pid
```

**New Flag**: `--force-kill` (kill active writer and proceed)

### 4. CLI Jobs Command

**Location**: `slavv_python/interface/cli/jobs.py`

**New CLI Entry**: `slavv jobs <subcommand>`

**Subcommands**:
- `slavv jobs list` - Show active jobs (table format)
- `slavv jobs history [--run-dir PATH]` - Show completed jobs
- `slavv jobs kill <job-id>` - Terminate a running job
- `slavv jobs daemon status` - Show daemon PID and uptime
- `slavv jobs daemon restart` - Restart monitoring daemon

**Example Output**:
```
$ slavv jobs list

Active Parity Jobs:
┌──────────────────────┬───────┬─────────────────────┬────────┬─────────┬─────────────────────┬──────────┐
│ Job ID               │ PID   │ Run Directory       │ Stage  │ Status  │ Started             │ Duration │
├──────────────────────┼───────┼─────────────────────┼────────┼─────────┼─────────────────────┼──────────┤
│ a3f2-41bd-9c8e-7d1f  │ 25248 │ .../crop_M_exact   │ energy │ running │ 2026-06-09 08:30:15 │ 2h 15m   │
└──────────────────────┴───────┴─────────────────────┴────────┴─────────┴─────────────────────┴──────────┘

$ slavv jobs history --run-dir workspace/runs/oracle_180709_E/crop_M_exact

Job History for: workspace/runs/oracle_180709_E/crop_M_exact
┌──────────────────────┬───────┬─────────────────────┬────────┬───────┬──────────┬──────────┐
│ Job ID               │ PID   │ Started             │ Stage  │ Exit  │ Duration │ Status   │
├──────────────────────┼───────┼─────────────────────┼────────┼───────┼──────────┼──────────┤
│ b7c1-51ea-8f3d-9a2c  │ 30880 │ 2026-06-05 14:20:03 │ energy │ 0     │ 4h 12m   │ success  │
│ a3f2-41bd-9c8e-7d1f  │ 25248 │ 2026-06-09 08:30:15 │ energy │ -     │ 2h 15m   │ running  │
└──────────────────────┴───────┴─────────────────────┴────────┴───────┴──────────┴──────────┘
```

## Integration with Existing Metadata

The system respects existing conventions:
- Reads `99_Metadata/parity_job.{pid,json}` for progress updates
- Does NOT replace these files (still written by `parity_experiment.py`)
- Uses these as progress checkpoints during monitoring

**Metadata Schema** (existing `parity_job.json`):
```json
{
  "pid": 25248,
  "command": "resume-exact-run ...",
  "started_at": "2026-06-09T08:30:15",
  "run_dir": "workspace/runs/oracle_180709_E/crop_M_exact",
  "oracle_root": "workspace/oracles/180709_E_crop_M",
  "stage": "energy",
  "last_checkpoint": "checkpoint_energy_octave_6.pkl"
}
```

## Error Handling

### Duplicate Writer Detection
```python
# Before starting any parity experiment
active_job = registry.get_job_by_run_dir(run_dir)
if active_job and is_process_alive(active_job.pid):
    raise DuplicateWriterError(
        f"Active writer detected:\n"
        f"  PID: {active_job.pid}\n"
        f"  Started: {active_job.started_at}\n"
        f"  Duration: {format_duration(now - active_job.started_at)}\n"
        f"\nOptions:\n"
        f"  1. Wait for job to complete\n"
        f"  2. Use --force-kill to terminate and proceed\n"
        f"  3. Check status: slavv jobs list"
    )
```

### Daemon Crash Recovery
- Daemon writes heartbeat to `workspace/scratch/monitor_daemon_heartbeat.json`
- If daemon crashes, next monitored job start detects stale heartbeat and restarts daemon
- Orphaned jobs (no daemon, but PID still alive) are re-registered on daemon restart

### Notification Failures
- If `win10toast` unavailable, log notification to `monitor_daemon.log` only
- Do not fail job execution if notifications fail

## Security Considerations

- **PID Reuse**: Validate process name matches expected Python interpreter
- **File Locking**: Use `fasteners.InterProcessLock` to prevent registry corruption
- **Path Validation**: Canonicalize all paths before comparison
- **Command Injection**: Never shell-execute user input; use subprocess array form

## Performance Impact

- Registry append operations: O(1) write
- Active job queries: O(n) scan of active jobs only (typically < 10)
- Daemon CPU: negligible (30s poll interval, lightweight checks)
- Disk I/O: ~1KB per job record, append-only

## Migration Path

### Phase 1: Core Infrastructure (This Spec)
- JobRegistry implementation
- MonitorDaemon implementation
- CLI integration (`--monitor` flag)
- `slavv jobs` commands

### Phase 2: Enhanced Features (Future)
- Slack/email notifications (opt-in via config)
- Streamlit dashboard integration
- Distributed monitoring (multiple machines)
- Resource usage tracking (CPU, memory, disk I/O)

## Testing Strategy

### Unit Tests
- `test_job_registry.py` - CRUD operations, locking, history queries
- `test_monitor_daemon.py` - Polling logic, notification dispatch, cleanup
- `test_cli_jobs.py` - Command output formatting, error handling

### Integration Tests
- `test_monitored_parity_run.py` - End-to-end: start → monitor → notify → complete
- `test_duplicate_writer_prevention.py` - Verify blocking behavior
- `test_daemon_recovery.py` - Crash and restart scenarios

### Manual Testing
- Windows 10/11 desktop notifications
- Cross-terminal session persistence
- Long-running job (4+ hours) notification delivery

## Dependencies

**New**:
- `win10toast` - Windows desktop notifications (pip install)
- `fasteners` - File-based locking (already used elsewhere in project)
- `tabulate` - CLI table formatting (already used elsewhere)

**Standard Library**:
- `subprocess`, `psutil`, `json`, `pathlib`, `datetime`, `uuid`, `logging`

## Configuration

**Config File**: `workspace/scratch/monitor_config.json` (optional)

```json
{
  "poll_interval_seconds": 30,
  "notification_enabled": true,
  "auto_archive_days": 30,
  "daemon_idle_timeout_minutes": 60
}
```

## Open Issues

1. **Cross-Platform**: Design assumes Windows; need Linux/macOS notification fallback
2. **Registry Growth**: JSONL will grow unbounded; need rotation/archival strategy
3. **Daemon Robustness**: Should daemon auto-restart on crash, or require manual start?

## Alternatives Considered

### Alternative 1: Systemd/Windows Service
**Pros**: Native OS integration, automatic restart
**Cons**: Complex setup, requires admin privileges, overkill for single-user dev workflow

### Alternative 2: Cron/Task Scheduler
**Pros**: No persistent daemon process
**Cons**: Cannot provide immediate notifications, 1-minute minimum poll interval

### Alternative 3: Streamlit Dashboard Only
**Pros**: Reuses existing UI infrastructure
**Cons**: Requires browser open, no desktop notifications, not scriptable

**Decision**: Simple Python daemon with JSONL registry provides best balance of simplicity and functionality for single-user dev environment.
