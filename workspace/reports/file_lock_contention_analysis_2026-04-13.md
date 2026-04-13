# File Lock Contention Incident Analysis (2026-04-13)

## Scope

This document explains the file lock contention issue observed during MATLAB/Python parity workflows in the canonical run root:

- `C:\slavv_comparisons\release_verify_20260413\live_canonical_20260413`

It captures symptoms, likely causes, impact, and an operational recovery/prevention runbook.

## Executive Summary

A rerun of the canonical parity command reported MATLAB failure almost immediately, with stderr messages indicating file access conflicts:

- `The process cannot access the file because it is being used by another process.`

At the same time, the run root already contained a complete MATLAB batch (`batch_260413-144432`) and successful resume metadata (`matlab_resume_state.json` status `completed`). This means the failure was not due to missing data; it was a coordination/locking conflict during rerun execution and/or log/metadata updates.

## What File Lock Contention Means Here

On Windows, if one process has an open handle with restrictive sharing flags, another process may be blocked from opening/writing/deleting the same file.

In this workflow, lock contention can occur when these actors overlap on the same run root:

1. `compare_matlab_python.py`
2. `run_matlab_cli.bat`
3. MATLAB process writing batch artifacts
4. Python process writing checkpoints/exports
5. manifest/status writers reading/writing metadata files

## Observed Evidence

### 1) Canonical run root metadata and manifests

- `C:\slavv_comparisons\release_verify_20260413\live_canonical_20260413\99_Metadata\run_manifest.md`
- `C:\slavv_comparisons\release_verify_20260413\live_canonical_20260413\99_Metadata\matlab_status.json`

Notable details:

- Preflight passed.
- `matlab_resume_mode` eventually reports `complete-noop`.
- `matlab_last_completed_stage` reports `network`.
- batch folder exists and is complete.

### 2) MATLAB batch/log indicates successful full completion

- `C:\slavv_comparisons\release_verify_20260413\live_canonical_20260413\01_Input\matlab_results\matlab_run.log`

Notable details:

- Energy, vertices, edges, and network stages completed.
- `Vectorization completed successfully!`
- Timing JSON written under batch folder.

### 3) Rerun command stderr showed lock conflict

During a subsequent canonical rerun, the CLI reported MATLAB exit code `1` with repeated stderr lines:

- `The process cannot access the file because it is being used by another process.`

This pattern is consistent with a file-handle conflict rather than a pure algorithmic/parity failure.

## Why This Happened (Likely Root Cause)

Most likely: a rerun was launched while artifacts and logs under the same run root were still being handled by another process (or rapidly re-opened by multiple workflow components). The rerun attempted to write/read files that were still locked.

Contributing factors:

1. Reusing a single output root for multiple sequential attempts with mixed failure/success states.
2. Running full orchestration (`--input ... --output-dir ...`) again after a complete MATLAB batch already existed.
3. Windows file-lock semantics being stricter than POSIX for concurrent writes and some read/write combinations.

## Impact Assessment

1. False-negative run status in some generated summaries/manifests during rerun windows.
2. MATLAB stage reports can look contradictory across artifacts if rerun fails before metadata converges.
3. Delays release verification and parity audit steps if operators rerun blindly against the same root.

## Recovery Runbook (Recommended)

### Preferred recovery path (already validated)

1. Do not relaunch MATLAB immediately when batch is complete.
2. Recompute comparison in standalone mode using completed artifacts:

```powershell
python workspace/scripts/cli/compare_matlab_python.py \
  --standalone-matlab-dir C:\slavv_comparisons\release_verify_20260413\live_canonical_20260413\01_Input\matlab_results \
  --standalone-python-dir C:\slavv_comparisons\release_verify_20260413\live_canonical_20260413\02_Output\python_results \
  --output-dir C:\slavv_comparisons\release_verify_20260413\live_canonical_20260413 \
  --comparison-depth deep
```

3. Verify final artifacts:

- `comparison_report.json`
- `summary.txt`
- `99_Metadata/run_manifest.md`
- `99_Metadata/matlab_status.json`

### If MATLAB must be relaunched

1. Ensure no stale MATLAB/process handles remain.
2. Use a fresh output root for each new launch attempt.
3. Avoid overlapping runs against the same root.

## Prevention Controls

### Operational controls

1. One-writer rule: only one active parity run per output root.
2. Fresh-root policy for new launch attempts (timestamped run roots).
3. If a batch is complete, switch to standalone analysis mode instead of rerunning MATLAB.

### Workflow controls

1. Gate rerun behavior on `matlab_status.json`:
   - if `matlab_batch_complete=true`, default to analysis-only path.
2. Add a lock/lease marker under `99_Metadata/` to signal active orchestration.
3. Add short retry/backoff for metadata writes where safe.

## Release Checklist Guidance

For release readiness, treat this as an operational execution issue (not a parity algorithm regression) when:

1. MATLAB batch artifacts indicate complete stage progression.
2. Standalone comparison can be generated from existing artifacts.
3. Preflight and resume semantics files are present and interpretable.

## Quick Triage Checklist

1. Inspect `99_Metadata/matlab_status.json` for completion and resume mode.
2. Inspect `01_Input/matlab_results/matlab_run.log` tail for full-stage completion.
3. If complete, run standalone comparison instead of relaunching MATLAB.
4. If incomplete, rerun with a fresh output root and no concurrent processes.

## Conclusion

The incident is best classified as file access contention during rerun orchestration on Windows, not a direct model/parity logic failure. The established safe path is to avoid redundant MATLAB relaunches once a batch is complete and use standalone comparison generation for final reports.
