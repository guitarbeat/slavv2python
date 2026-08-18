---
title: Detached Exact-Route Parity Jobs
module: analytics/parity
tags: [parity, exact-route, run-monitoring, windows]
problem_type: workflow
resolution_type: runbook
---

# Detached Exact-Route Parity Jobs

## In short

Overnight jobs must outlive the chat session. Launch them detached so the run
folder owns the PID and logs. A leftover PID file is not proof the writer is
alive.

## Problem
Long exact-route parity reruns were launched from an interactive Codex session and tracked by scratch PID files. If the agent session ended or the process was terminated, the run snapshot could remain `running` while no writer was alive and no checkpoint was produced.

## Evidence
The `crop_M_exact` energy proof failed before comparison because `02_Output/python_results/checkpoints/checkpoint_energy.pkl` was missing. The snapshot still reported `energy` as running, while `workspace/scratch/crop_energy_rerun_latest.pid` pointed at dead PID `31796`.

## Root Cause
The operating system did not own a durable parity job record under the run root. Monitoring depended on agent-side memory plus a scratch PID file, so interruption left stale state and no run-local manifest/log surface.

## Solution
Launch long exact-route reruns with `launch-exact-run` so the process is detached and all operator artifacts live under `99_Metadata/`:

```powershell
slavv parity launch-exact-run `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M_v2 `
  --force-rerun-from energy `
  --stop-after energy `
  --skip-preflight `
  --n-jobs 3
```

Monitor from any later shell or agent session:

```powershell
slavv parity status-exact-run `
  --run-dir workspace/runs/oracle_180709_E/crop_M_exact
```

Use run-local artifacts first: `99_Metadata/parity_job.json`, `parity_job.pid`, `parity_job.out.log`, and `parity_job.err.log`. Treat scratch PID files as legacy fallbacks.

## Verification
Unit coverage passed for the detached launcher and monitor discovery:

```powershell
python -m pytest tests/unit/interface/test_monitor_service.py tests/unit/parity/test_parity_experiment_comprehensive.py -q
```

Result: `13 passed`.

The live crop rerun launched as run-local parity job PID `25248`, and `status-exact-run` reported `Effective status: running (PID 25248 is alive.)`.

## Follow-Up
After the detached energy rerun exits, run `prove-exact --stage energy`. Continue to vertices/network refresh and `prove-exact-sequence` only if energy reaches strict zero.

## Addendum (2026-08-13): Windows `launch-exact-run` lease suicide

`slavv parity launch-exact-run` is **not** a safe launcher on this Windows box.

**Evidence:** two launches of `canonical_full_v18` wrote `writer_lease.json` with the `Popen` PID, then the detached `resume-exact-run` child died immediately:

```text
RuntimeError: Run directory has active writer lease (PID 30528). Use --force-kill to replace it.
```

`--force-kill` on a second `launch-exact-run` still failed. `slavv jobs list` hung (multiple live `jobs list` processes). Windows then reused PID `23328`; a `--monitor` resume treated the ghost registry job as a live writer.

**Root cause:** `launch_exact_run_job` writes the lease as the child PID, then the child `resume_writer_session` reconciles that lease as a *different* live Python process. Closing the parent's redirected stdio handles under `DETACHED_PROCESS` is an additional Windows foot-gun.

**Do this instead:**

1. Do **not** wait on `slavv jobs list`. Read every `99_Metadata/writer_lease.json` and check `Get-Process -Id <pid>`.
2. Seed a **new** dest: copy Energy/Vertices/params/provenance only. Exclude `04_Edges`, `05_Network`, `checkpoint_edges.pkl`, `checkpoint_network.pkl`, `03_Analysis` proofs, and old `writer_lease` / `parity_job.*`.
3. `preflight-exact` the dest. Pass `--dataset-root` — `00_Refs` may be empty.
4. Clear ghost registry rows for that dest (`JobRegistry.update_job(..., status="killed")`).
5. Start the writer with `Start-Process` on `resume-exact-run` (not `launch-exact-run`). `--force-kill` is fine; add `--monitor` only after the lease PID is the new live process.

```powershell
Start-Process -FilePath .\.venv\Scripts\python.exe -ArgumentList @(
  "-m","slavv_python.interface.cli.parity","resume-exact-run",
  "--dest-run-root","workspace/runs/oracle_180709_E/canonical_full_v18",
  "--dataset-root","workspace/datasets/771eb62fd1322cf59e24f056aff2692b3375b94ce6dc9b25744428d4dbf1e353",
  "--oracle-root","workspace/oracles/180709_E_full_v2",
  "--force-rerun-from","edges",
  "--stop-after","network",
  "--skip-preflight",
  "--force-kill"
) -WorkingDirectory (Get-Location) -WindowStyle Hidden `
  -RedirectStandardOutput "workspace/runs/oracle_180709_E/canonical_full_v18/99_Metadata/parity_job.out.log" `
  -RedirectStandardError "workspace/runs/oracle_180709_E/canonical_full_v18/99_Metadata/parity_job.err.log" `
  -PassThru
```

**Verified:** lease PID 20564 alive; snapshot `running edges` Watershed Discovery after cached Energy/Vertices load. Watch with `status-exact-run`, not `jobs list`.
