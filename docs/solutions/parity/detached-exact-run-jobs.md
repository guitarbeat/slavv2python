---
title: Detached Exact-Route Parity Jobs
module: analytics/parity
tags: [parity, exact-route, run-monitoring, windows]
problem_type: workflow
resolution_type: runbook
---

# Detached Exact-Route Parity Jobs

## Problem
Long exact-route parity reruns were launched from an interactive Codex session and tracked by scratch PID files. If the agent session ended or the process was terminated, the run snapshot could remain `running` while no writer was alive and no checkpoint was produced.

## Evidence
The `crop_M_exact` energy proof failed before comparison because `02_Output/python_results/checkpoints/checkpoint_energy.pkl` was missing. The snapshot still reported `energy` as running, while `workspace/scratch/crop_energy_rerun_latest.pid` pointed at dead PID `31796`.

## Root Cause
The operating system did not own a durable parity job record under the run root. Monitoring depended on agent-side memory plus a scratch PID file, so interruption left stale state and no run-local manifest/log surface.

## Solution
Launch long exact-route reruns with `launch-exact-run` so the process is detached and all operator artifacts live under `99_Metadata/`:

```powershell
python scripts/parity_experiment.py launch-exact-run `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M `
  --force-rerun-from energy `
  --stop-after energy `
  --skip-preflight `
  --n-jobs 3
```

Monitor from any later shell or agent session:

```powershell
python scripts/parity_experiment.py status-exact-run `
  --run-dir workspace/runs/oracle_180709_E/crop_M_exact
```

Use run-local artifacts first: `99_Metadata/parity_job.json`, `parity_job.pid`, `parity_job.out.log`, and `parity_job.err.log`. Treat scratch PID files as legacy fallbacks.

## Verification
Unit coverage passed for the detached launcher and monitor discovery:

```powershell
python -m pytest tests/unit/apps/test_monitor_service.py tests/unit/scripts/parity_experiment/test_parity_experiment_comprehensive.py -q
```

Result: `13 passed`.

The live crop rerun launched as run-local parity job PID `25248`, and `status-exact-run` reported `Effective status: running (PID 25248 is alive.)`.

## Follow-Up
After the detached energy rerun exits, run `prove-exact --stage energy`. Continue to vertices/network refresh and `prove-exact-sequence` only if energy reaches strict zero.
