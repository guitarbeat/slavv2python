# Requirements: Automated Parity Job Monitoring System

## Problem Statement

Long-running parity experiments (energy reruns, prove-exact-sequence) can take 4+ hours to complete. Currently, developers must:
- Manually check PID files in `workspace/scratch/` and `99_Metadata/`
- Periodically run `slavv monitor` or `status-exact-run` commands
- Risk starting duplicate writers on the same run directory
- Miss completion/failure notifications until they manually check back

This creates inefficiency and increases the risk of workflow errors.

## Goals

1. **Automated Status Tracking**: System automatically monitors active parity jobs
2. **Completion Notification**: Alert when jobs complete (success or failure)
3. **Conflict Prevention**: Block duplicate writers on active run directories
4. **Historical Logging**: Maintain audit trail of all parity job executions
5. **Cross-Session Persistence**: Monitor jobs across terminal/IDE restarts

## Success Criteria

- Developer starts a long parity job and walks away
- System notifies via desktop notification + log file when job completes
- Attempting to start a duplicate job on the same run-dir fails with clear message
- Job history is queryable (when started, duration, exit code, last status)
- Works on Windows (primary development environment)

## Non-Goals

- Real-time Streamlit/web dashboard (already exists via `slavv-app`)
- Distributed/remote job orchestration
- Log streaming/tailing (use `slavv monitor` for interactive watching)
- Email/Slack notifications (desktop only)

## Constraints

- Must integrate with existing `parity_experiment.py` CLI
- Must respect run-local `99_Metadata/parity_job.{pid,json}` files
- No new runtime dependencies beyond stdlib + existing project deps
- Must work in Windows PowerShell environment
- Cannot break existing detached job workflows

## User Stories

### Story 1: Start monitored job
```powershell
slavv parity resume-exact-run \
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact \
  --oracle-root workspace/oracles/180709_E_crop_M \
  --force-rerun-from energy \
  --stop-after network \
  --skip-preflight \
  --monitor
```
**Expected**: Job runs detached, monitoring daemon starts, desktop notification shows "Job started"

### Story 2: Check job status
```powershell
slavv jobs list
# Output:
# PID    | Run Dir                                      | Stage  | Status  | Started            | Duration
# 25248  | workspace/runs/.../crop_M_exact             | energy | running | 2026-06-09 08:30  | 2h 15m
```

### Story 3: Job completes
**Expected**: Desktop notification: "Parity job 25248 completed successfully (energy stage) - Duration: 4h 12m"

### Story 4: Prevent duplicate writer
```powershell
slavv parity resume-exact-run \
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact \
  ...
```
**Expected**: Error before execution: "Run directory has active writer (PID 25248). Use --force-kill or wait for completion."

### Story 5: Query job history
```powershell
slavv jobs history --run-dir workspace/runs/oracle_180709_E/crop_M_exact
# Output:
# PID    | Started            | Completed          | Duration | Exit Code | Stage
# 30880  | 2026-06-05 14:20  | 2026-06-05 18:32  | 4h 12m   | 0         | energy
# 25248  | 2026-06-09 08:30  | [running]         | 2h 15m   | -         | energy
```

## Acceptance Criteria

- [ ] `parity_experiment.py` accepts `--monitor` flag for detached runs
- [ ] Monitoring daemon persists across IDE/terminal restarts
- [ ] Desktop notifications work on Windows 10/11
- [ ] `slavv jobs list` shows all active monitored jobs
- [ ] `slavv jobs history` shows completed jobs from persistent log
- [ ] Attempting duplicate run on active directory fails with clear error
- [ ] Job metadata includes: PID, run-dir, start time, stage, oracle path
- [ ] Daemon logs to `workspace/scratch/job_monitor.log`
- [ ] Job history persists to `workspace/scratch/job_history.jsonl`
- [ ] Works with existing `99_Metadata/parity_job.{pid,json}` convention

## Open Questions

1. **Notification Library**: Use `win10toast` (pure Python) or PowerShell `New-BurnerToastNotification`?
2. **Daemon Process**: Separate Python daemon or integrate into existing `parity_experiment.py`?
3. **Health Checks**: Poll interval for checking job liveness (default: 30s)?
4. **Cleanup**: Auto-archive completed job metadata after N days?

## References

- [Exact Proof Findings](../../docs/reference/core/EXACT_PROOF_FINDINGS.md) - Cold-start protocol section
- [Detached Exact-Run Jobs Solution](../../docs/solutions/parity/detached-exact-run-jobs.md)
- Current PID tracking: `workspace/scratch/crop_energy_rerun_latest.pid`, `99_Metadata/parity_job.pid`
