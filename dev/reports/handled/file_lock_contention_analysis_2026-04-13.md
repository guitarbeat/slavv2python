# File Lock Contention Incident Analysis (2026-04-13)

## What this file is for

This is the canonical April 13 release-operations incident note. It combines
the useful timeline from the release attempt with the later lock-contention
analysis into one operator-focused runbook.

## Read this when

- a parity rerun fails on Windows with file-access errors
- you need to know whether a failure is an operational incident or a parity bug
- you need the safe recovery path for a run root that already contains a
  completed MATLAB batch
- you need the April 13 release execution timeline in one place

## Executive Summary

- April 13 setup work succeeded: the diagnostic setup gate, output-root
  preflight, baseline quality gate, and MATLAB health check all passed.
- A fallback live run on `skimage.data.multipage.tif` is not release-grade
  evidence. It showed a tiny-input energy failure and zero graph outputs, and
  it should not be used for release-readiness claims.
- The later canonical run root
  `C:\slavv_comparisons\release_verify_20260413\live_canonical_20260413`
  already contained a completed MATLAB batch and `complete-noop` resume status.
- A subsequent rerun then failed with repeated Windows file-access errors:
  `The process cannot access the file because it is being used by another process.`
- This is best classified as rerun orchestration/file-lock contention, not as a
  parity algorithm regression.
- The safe recovery path is to stop relaunching MATLAB and generate the final
  comparison from the existing MATLAB and Python artifacts in standalone mode.

## Handling Classification

- Status: Handled
- Why: This document records a resolved operations incident with a stable
  recovery path and prevention checklist.
- Open algorithmic parity work is tracked separately in
  `dev/reports/unhandled/` and is not reopened by this report.

## Current Status

### Confirmed and usable

- The environment/setup gates passed on April 13.
- The MATLAB health check passed in about `43s`.
- The canonical run root contains a completed MATLAB batch and interpretable
  metadata.
- A safe standalone-analysis recovery path exists for completed runs.

### Active risk

- Reusing the same output root for another full orchestration attempt can
  produce false-negative failure states or contradictory metadata while handles
  are still active.
- Windows file-lock semantics make same-root reruns riskier than a fresh-root
  launch or standalone analysis from completed artifacts.

### Superseded conclusions

- The fallback `multipage.tif` run is not evidence of release readiness or
  parity behavior on canonical data.
- A rerun failure on this completed canonical root should not be treated as an
  automatic parity regression.

## April 13 execution timeline

1. Diagnostic setup gate passed:
   `python -m pytest dev/tests/diagnostic/test_comparison_setup.py`
2. Output-root preflight passed against
   `C:\slavv_comparisons\release_verify_20260413`.
3. Baseline quality gate passed:
   `compileall`, `ruff format --check`, `ruff check`, `mypy`, and
   `pytest -m "unit or integration"`.
4. MATLAB health check passed.
5. A fallback live run at
   `C:\slavv_comparisons\release_verify_20260413\live_20260413b` produced
   staged artifacts, but the tiny fallback TIFF caused a MATLAB energy-stage
   dimension mismatch and both sides produced zero graph outputs.
6. A later canonical run root completed MATLAB successfully.
7. A subsequent rerun against that canonical root hit file-lock contention.

## Classification and evidence

The later rerun should be classified as a Windows coordination issue because
the authoritative artifacts already indicated successful completion:

- `99_Metadata/matlab_status.json` reported `matlab_resume_mode=complete-noop`
  and `matlab_last_completed_stage=network`
- the batch folder existed and was complete
- `01_Input/matlab_results/matlab_run.log` showed successful energy, vertices,
  edges, and network completion, including `Vectorization completed successfully!`
- the rerun stderr showed repeated file-access errors instead of a parity logic
  mismatch

That evidence points to a completed batch plus an unsafe same-root rerun, not a
missing-data or algorithmic failure.

## Safe recovery path

If the batch is already complete, do not relaunch MATLAB. Recompute the final
comparison from the completed artifacts:

```powershell
python dev/scripts/cli/compare_matlab_python.py \
  --standalone-matlab-dir C:\slavv_comparisons\release_verify_20260413\live_canonical_20260413\01_Input\matlab_results \
  --standalone-python-dir C:\slavv_comparisons\release_verify_20260413\live_canonical_20260413\02_Output\python_results \
  --output-dir C:\slavv_comparisons\release_verify_20260413\live_canonical_20260413 \
  --comparison-depth deep
```

Then verify:

- `03_Analysis/comparison_report.json`
- `03_Analysis/summary.txt`
- `99_Metadata/run_manifest.md`
- `99_Metadata/matlab_status.json`

## Rerun prevention checklist

1. Check `99_Metadata/matlab_status.json` before relaunching anything.
2. If `matlab_batch_complete=true` or the resume mode is `complete-noop`, do
   standalone analysis instead of a full MATLAB relaunch.
3. Enforce a one-writer rule per output root.
4. Use a fresh output root for any genuinely new MATLAB launch attempt.
5. If a relaunch is unavoidable, make sure no stale MATLAB or workflow handles
   still target the same files.

## Operator guidance

Treat this as an operational execution issue, not a parity regression, when all
three of the following are true:

1. the batch/log artifacts show full MATLAB stage completion
2. the resume metadata is present and interpretable
3. standalone comparison can be generated from the existing artifacts

If those conditions are not met, then fall back to the normal preflight logic
and use a fresh output root for the next launch attempt.

