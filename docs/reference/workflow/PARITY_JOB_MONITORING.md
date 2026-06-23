# Parity Job Monitoring

[Up: Reference Docs](../README.md) · [Parity Pre-Gate](PARITY_PRE_GATE.md) · [Parity Certification Guide](PARITY_CERTIFICATION_GUIDE.md)

Comprehensive guide to the automated parity job monitoring system for tracking long-running experiments across terminal sessions.

---

## Overview

The Parity Job Monitoring System automates the tracking of long-running parity experiments (energy reruns, prove-exact-sequence) that can take 4+ hours to complete. It provides:

- **Automatic job registration** when using the `--monitor` flag
- **Background monitoring daemon** that persists across terminal/IDE restarts
- **Desktop notifications** on job completion/failure (Windows)
- **Duplicate writer prevention** to avoid concurrent writes to the same run directory
- **Job history persistence** for audit trails and debugging
- **CLI commands** for monitoring active jobs and viewing history

---

## Architecture

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
**Last Updated:** 2026-06-10

---

#### JobRegistry (`analytics/parity/job_registry.py`)
- Persistent JSONL storage at `workspace/scratch/job_registry.jsonl`
- File-based locking via `fasteners.InterProcessLock`
- CRUD operations for job records
- **Timestamp Refresh**: `update_job` always refreshes `last_seen_at` to ensure the most recent record is correctly identified by the monitor daemon.
- Query interface for active jobs and history

#### MonitorDaemon (`analytics/parity/monitor_daemon.py`)
- Background process monitoring job registry every 30 seconds
- Writes PID to `workspace/scratch/monitor_daemon.pid`
- Logs to `workspace/scratch/monitor_daemon.log`
- Sends desktop notifications via `win10toast` (Windows)
- Auto-terminates after 60 minutes of idle time

#### Process Utilities (`analytics/parity/process_utils.py`)
- Process liveness checking with `psutil`
- PID reuse protection (validates process name)
- Process tree termination

---

## Usage

> On Windows, activate `.venv` before using these commands. For direct Python
> invocations use `./.venv/Scripts/python.exe`; the ambient `python` executable
> is not a supported parity-test runner.

### Starting a Monitored Job

Add the `--monitor` flag to any `resume-exact-run` or `launch-exact-run` command:

```powershell
# Start monitored parity job
slavv parity resume-exact-run \
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact \
  --oracle-root workspace/oracles/180709_E_crop_M \
  --force-rerun-from energy \
  --stop-after network \
  --skip-preflight \
  --monitor
```

**What happens:**
1. System checks for active writers on the run directory
2. If none found, starts detached subprocess
3. Registers job in persistent registry
4. Starts monitoring daemon (if not already running)
5. Desktop notification: "Parity job started (energy stage)"
6. Terminal can be closed safely

### Viewing Active Jobs

```powershell
slavv jobs list
```

**Example output:**
```
Active Parity Jobs:
┌──────────────────────┬───────┬─────────────────────┬────────┬─────────┬─────────────────────┬──────────┐
│ Job ID               │ PID   │ Run Directory       │ Stage  │ Status  │ Started             │ Duration │
├──────────────────────┼───────┼─────────────────────┼────────┼─────────┼─────────────────────┼──────────┤
│ a3f2-41bd-9c8e-7d1f  │ 25248 │ .../crop_M_exact   │ energy │ running │ 2026-06-09 08:30:15 │ 2h 15m   │
└──────────────────────┴───────┴─────────────────────┴────────┴─────────┴─────────────────────┴──────────┘
```

### Viewing Job History

```powershell
# All jobs
slavv jobs history

# Jobs for specific run directory
slavv jobs history --run-dir workspace/runs/oracle_180709_E/crop_M_exact

# Limit results
slavv jobs history --limit 10
```

**Example output:**
```
Job History for: workspace/runs/oracle_180709_E/crop_M_exact
┌──────────────────────┬───────┬─────────────────────┬────────┬───────┬──────────┬──────────┐
│ Job ID               │ PID   │ Started             │ Stage  │ Exit  │ Duration │ Status   │
├──────────────────────┼───────┼─────────────────────┼────────┼───────┼──────────┼──────────┤
│ b7c1-51ea-8f3d-9a2c  │ 30880 │ 2026-06-05 14:20:03 │ energy │ 0     │ 4h 12m   │ success  │
│ a3f2-41bd-9c8e-7d1f  │ 25248 │ 2026-06-09 08:30:15 │ energy │ -     │ 2h 15m   │ running  │
└──────────────────────┴───────┴─────────────────────┴────────┴───────┴──────────┴──────────┘
```

### Killing a Job

```powershell
# Kill by job ID (accepts partial IDs)
slavv jobs kill a3f2

# Kill by full ID
slavv jobs kill a3f2-41bd-9c8e-7d1f
```

The system will:
1. Terminate the process and all children
2. Update job status to 'killed'
3. Send desktop notification

### Managing the Daemon

```powershell
# Check daemon status
slavv jobs daemon status
```

**Example output:**
```
Monitor Daemon Status:
  PID: 30124
  Status: Running
  Uptime: 3h 45m
  Active Jobs: 2
  Last Heartbeat: 2026-06-09 14:23:15 (5 seconds ago)
```

```powershell
# Restart daemon
slavv jobs daemon restart
```

The daemon automatically:
- Starts when the first monitored job is registered
- Self-terminates after 60 minutes with no active jobs
- Restarts when a new monitored job is registered

---

## Duplicate Writer Prevention

The monitoring system prevents concurrent writes to the same run directory:

```powershell
# First job starts successfully
slavv parity resume-exact-run \
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact \
  --oracle-root workspace/oracles/180709_E_crop_M \
  --monitor

# Second attempt fails with clear error
slavv parity resume-exact-run \
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact \
  --oracle-root workspace/oracles/180709_E_crop_M \
  --monitor

# Error output:
# RuntimeError: Run directory has active writer (PID 25248).
# Job started: 2026-06-09 08:30:15
# Duration: 2h 15m
# 
# Options:
#   1. Wait for job to complete
#   2. Use --force-kill to terminate and proceed
#   3. Check status: slavv jobs list
```

### Force Killing Active Writer

If you need to replace an active job:

```powershell
slavv parity resume-exact-run \
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact \
  --oracle-root workspace/oracles/180709_E_crop_M \
  --force-kill \
  --monitor
```

**What happens:**
1. Terminates existing job (PID 25248)
2. Updates old job status to 'killed'
3. Starts new job
4. Registers new job with fresh ID
5. Desktop notification for both termination and start

---

## Desktop Notifications

### Windows 10/11

The system uses `win10toast` to send native Windows toast notifications:

**Job Started:**
```
SLAVV Parity Job Started
Job a3f2... (PID 25248) - energy stage
Run: .../crop_M_exact
```

**Job Completed (Success):**
```
SLAVV Parity Job Completed
Job a3f2... finished successfully
Stage: energy | Duration: 4h 12m
Exit code: 0
```

**Job Failed:**
```
SLAVV Parity Job Failed
Job a3f2... failed
Stage: energy | Duration: 1h 3m
Exit code: 1
```

### Notification Fallback

If `win10toast` is unavailable (Linux, WSL, CI), notifications are logged only:

```
2026-06-09 14:30:15 [INFO] Job a3f2-41bd completed successfully (energy, 4h 12m)
```

---

## Job Record Schema

Each job record contains:

```python
{
    "job_id": "a3f2-41bd-9c8e-7d1f",         # UUID
    "pid": 25248,                             # Process ID
    "run_dir": "workspace/runs/.../crop_M_exact",
    "oracle_root": "workspace/oracles/180709_E_crop_M",
    "stage": "energy",                        # 'energy', 'vertices', 'edges', 'network', 'sequence'
    "command": "resume-exact-run ...",        # Full CLI command
    "started_at": "2026-06-09T08:30:15",     # ISO 8601
    "last_seen_at": "2026-06-09T12:45:30",   # Updated every 30s
    "completed_at": "2026-06-09T12:42:27",   # When job finished
    "exit_code": 0,                           # Process exit code
    "status": "running",                      # 'running', 'completed', 'failed', 'killed'
    "metadata": {                             # Extra context
        "flags": ["--force-rerun-from", "--monitor"],
        "stop_after": "network"
    }
}
```

---

## Integration with Existing Metadata

The monitoring system **respects** existing run-local metadata:

- Reads `99_Metadata/parity_job.pid` for PID tracking
- Reads `99_Metadata/parity_job.json` for progress updates
- Does **not** replace these files (still written by `parity_experiment.py`)
- Uses them as checkpoints during monitoring

**Metadata precedence:**
1. Run-local `99_Metadata/parity_job.{pid,json}` (authoritative)
2. JobRegistry JSONL (cross-session history)
3. Legacy scratch PID files (deprecated)

---

## Troubleshooting

### Daemon Not Starting

**Symptoms:** Jobs register but no notifications arrive

**Check daemon status:**
```powershell
slavv jobs daemon status
```

**Manually restart:**
```powershell
slavv jobs daemon restart
```

**Check logs:**
```powershell
Get-Content workspace/scratch/monitor_daemon.log -Tail 50
```

### Stale PID Detection

**Symptoms:** "Active writer detected" but no process running

**Cause:** PID reuse (old PID reassigned to new process)

**Solution:** System validates process name contains 'python'. If validation fails, job is marked stale. Use `--force-kill` to clear:

```powershell
slavv parity resume-exact-run \
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact \
  --oracle-root workspace/oracles/180709_E_crop_M \
  --force-kill \
  --monitor
```

### Notifications Not Appearing

**Symptoms:** Daemon running but no desktop notifications

**Check:**
1. Is `win10toast` installed? `pip show win10toast`
2. Are Windows notifications enabled? Settings → System → Notifications
3. Check daemon logs for notification attempts:
   ```powershell
   Select-String "notification" workspace/scratch/monitor_daemon.log
   ```

**Workaround:** Notifications are logged even if toast fails. Monitor logs directly.

### Job Registry Corruption

**Symptoms:** `slavv jobs list` fails with JSON decode error

**Cause:** Concurrent writes without lock (rare)

**Solution:**
1. Backup corrupted file:
   ```powershell
   Copy-Item workspace/scratch/job_registry.jsonl workspace/scratch/job_registry.jsonl.backup
   ```
2. Remove corrupted lines (open in text editor, delete invalid JSON)
3. Or start fresh (loses history):
   ```powershell
   Remove-Item workspace/scratch/job_registry.jsonl
   ```

### Daemon Consuming Resources

**Symptoms:** High CPU or memory usage

**Check:**
- Normal: <1% CPU, ~50MB memory
- Abnormal: >5% CPU sustained

**Solution:** Restart daemon:
```powershell
slavv jobs daemon restart
```

If issue persists, check logs for infinite loops or errors.

---

## Configuration

The system uses sensible defaults. For advanced users, behavior can be customized:

### Polling Interval

Default: 30 seconds

To change, edit `monitor_daemon.py`:
```python
POLL_INTERVAL_SECONDS = 30  # Change to desired interval
```

### Idle Timeout

Default: 60 minutes

To change, edit `monitor_daemon.py`:
```python
IDLE_TIMEOUT_MINUTES = 60  # Change to desired timeout
```

### Notification Duration

Default: 10 seconds

Toast notification display time (Windows):
```python
toaster.show_toast(
    title="...",
    msg="...",
    duration=10,  # Change to desired seconds
    threaded=True
)
```

---

## Best Practices

### When to Use `--monitor`

✅ **Use monitoring for:**
- Long parity experiments (>1 hour)
- Overnight runs
- Energy stage reruns (4+ hours)
- Sequential prove-exact-sequence
- Any detached job you want notification for

❌ **Don't use monitoring for:**
- Quick diagnostic runs (<5 minutes)
- Interactive `slavv monitor --run-dir` sessions
- CI/CD automated tests

### Cold-Start Protocol with Monitoring

Updated protocol from [EXACT_PROOF_FINDINGS.md](../core/EXACT_PROOF_FINDINGS.md):

1. **Check active jobs:**
   ```powershell
   slavv jobs list
   ```

2. **Check run-local metadata:**
   ```powershell
   slavv parity status-exact-run \
     --run-dir workspace/runs/oracle_180709_E/crop_M_exact
   ```

3. **If job active, don't start another writer**

4. **If job completed, run proof:**
   ```powershell
   slavv parity prove-exact \
     --source-run-root workspace/runs/oracle_180709_E/crop_M_exact \
     --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact \
     --oracle-root workspace/oracles/180709_E_crop_M \
     --stage energy
   ```

5. **If proof passes, refresh downstream:**
   ```powershell
   slavv parity resume-exact-run \
     --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact \
     --oracle-root workspace/oracles/180709_E_crop_M \
     --force-rerun-from vertices \
     --stop-after network \
     --skip-preflight \
     --monitor
   ```

### Monitoring Multiple Jobs

The system supports multiple concurrent monitored jobs:

```powershell
# Start crop harness (energy only)
slavv parity resume-exact-run \
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact \
  --oracle-root workspace/oracles/180709_E_crop_M \
  --force-rerun-from energy \
  --stop-after energy \
  --skip-preflight \
  --monitor

# Start canonical run (full pipeline) in parallel
slavv parity resume-exact-run \
  --dest-run-root workspace/runs/oracle_180709_E/phase1_cert_network \
  --oracle-root workspace/oracles/180709_E_batch_190910-103039 \
  --force-rerun-from energy \
  --stop-after network \
  --skip-preflight \
  --monitor
```

Both jobs tracked independently, notifications for each.

### Job History Maintenance

The registry grows indefinitely. Periodic cleanup recommended:

```powershell
# Manually archive old jobs (future feature)
# For now, backup and truncate:
Copy-Item workspace/scratch/job_registry.jsonl workspace/scratch/job_registry_backup_$(Get-Date -Format 'yyyyMMdd').jsonl

# Keep last 100 lines (recent history)
Get-Content workspace/scratch/job_registry.jsonl | Select-Object -Last 100 | Set-Content workspace/scratch/job_registry.jsonl
```

---

## Implementation Details

### File Locking Strategy

Uses `fasteners.InterProcessLock` for atomic JSONL operations:

```python
lock_path = registry_path.with_suffix('.lock')
with InterProcessLock(lock_path):
    # Read or write registry
    ...
```

Prevents corruption from concurrent `slavv jobs` queries and daemon updates.

### PID Reuse Protection

Process validation prevents false positives:

```python
def is_process_alive(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
        # Validate process name contains 'python'
        if 'python' not in proc.name().lower():
            return False
        return proc.is_running()
    except psutil.NoSuchProcess:
        return False
```

### Daemon Heartbeat

Daemon writes heartbeat every poll cycle:

```json
{
    "pid": 30124,
    "last_update": "2026-06-09T14:30:15",
    "active_jobs": 2,
    "uptime_seconds": 13500
}
```

Location: `workspace/scratch/monitor_daemon_heartbeat.json`

Used by `slavv jobs daemon status` for live status.

---

## Related Documentation

- **[Parity Pre-Gate](PARITY_PRE_GATE.md):** Three-tier parity workflow with monitoring examples
- **[Parity Certification Guide](PARITY_CERTIFICATION_GUIDE.md):** Full certification workflow
- **[Exact Proof Findings](../core/EXACT_PROOF_FINDINGS.md):** Live parity status and cold-start protocol
- **[Solution: Detached Exact-Run Jobs](../../solutions/parity/detached-exact-run-jobs.md):** Background for monitoring system design

---

## Future Enhancements

Potential improvements tracked in [TODO.md](../../TODO.md):

- Email/Slack notification plugins
- Streamlit dashboard integration
- Resource usage tracking (CPU, memory, disk I/O)
- Automatic log rotation
- Job history archival (auto-cleanup after N days)
- Multi-machine distributed monitoring

---

**Last Updated:** 2026-06-09
