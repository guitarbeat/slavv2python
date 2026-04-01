# Comparison Resume Transparency Implementation Plan

Status: Draft
Date: 2026-04-01

## Objective

Make comparison-mode MATLAB/Python runs answer these questions without manual
forensics:

1. Is the next rerun fresh, resumed, or a no-op?
2. If resumed, which batch is being reused?
3. Which stage will run next?
4. If the last run crashed mid-stage, will the rerun truly resume work or
   restart that stage?
5. Which files are authoritative for run status, failure cause, and restart
   behavior?
6. What exact evidence should a user inspect after a crash?

This plan focuses on transparency and predictability first. It does not assume
that the first implementation must also add new low-level MATLAB resume
behavior.

## Why This Matters

The current repo intent is already documented:

- [docs/COMPARISON_LAYOUT.md](COMPARISON_LAYOUT.md) says resumed runs and manual
  inspections should be predictable.
- [docs/PARITY_FINDINGS_2026-03-27.md](PARITY_FINDINGS_2026-03-27.md) records
  that fresh MATLAB reruns on `C:` failed during `energy`, while the same work
  succeeded on `D:`.
- The user complaint is not just that the run is slow. The deeper problem is
  that a crash leaves behind enough partial artifacts to imply progress, but not
  enough trustworthy status to answer whether a rerun will reuse that progress.

Healthy comparison runs are expensive enough that this ambiguity is costly:

- Healthy standalone MATLAB runs were roughly `743s` to `909s`, with `energy`
  dominating runtime.
- Healthy full native-Python runs were roughly `3281s` to `3386s`, again mostly
  in `energy`.
- Imported-MATLAB parity runs can cut the Python side down to about `4m 52s`,
  but only when MATLAB produced reusable `energy` and `vertices` outputs.

When rerun semantics are unclear, the operator cannot tell whether they are
about to pay for a full expensive stage again.

## Files Already Available

### Current failing and partial run artifacts

These files already exist and are the best evidence for the current issue:

- `comparison_results_test_run/99_Metadata/run_snapshot.json`
- `comparison_results_test_run/01_Input/matlab_results/matlab_resume_state.json`
- `comparison_results_test_run/01_Input/matlab_results/matlab_run.log`
- `comparison_results_test_run/01_Input/matlab_results/batch_260401-122735/`
- `comparison_results/99_Metadata/run_snapshot.json`
- `comparison_results/01_Input/matlab_results/matlab_resume_state.json`
- `comparison_results/01_Input/matlab_results/matlab_run.log`
- `comparison_results/01_Input/matlab_results/batch_260331-183658/`

Observed from these artifacts:

- the shared comparison snapshot can still say the MATLAB task is running
- the MATLAB-specific resume file can say `running:energy`
- the top-level MATLAB log can already have ended with a nonzero exit
- partial chunk files can exist inside the batch tree even though the MATLAB
  stage was not completed

That combination is exactly why the workflow feels like a black box.

### Existing documentation and reports

- `docs/COMPARISON_LAYOUT.md`
- `docs/PARITY_FINDINGS_2026-03-27.md`
- `workspace/reports/matlab_standalone_consistency_2026-03-28.md`
- `workspace/reports/python_matlab_parity_postfix_2026-03-30.md`
- `workspace/reports/python_standalone_consistency_postfix_2026-03-30.md`

### Core code paths involved

- `workspace/scripts/cli/run_matlab_cli.bat`
- `workspace/scripts/cli/run_matlab_vectorization.m`
- `source/slavv/evaluation/comparison.py`
- `source/slavv/runtime/run_state.py`
- `source/slavv/apps/cli.py`
- `source/slavv/evaluation/management.py`
- `source/slavv/core/pipeline.py`
- `source/slavv/core/energy.py`
- `external/Vectorization-Public/source/get_energy_V202.m`
- `external/Vectorization-Public/source/mat2h5.m`

### Relevant existing tests

- `tests/unit/analysis/test_comparison_runtime.py`
- `tests/unit/runtime/test_run_state.py`

## Current Behavior

### 1. There are two separate run ledgers

The current comparison workflow has two different status systems:

- a shared structured run snapshot under `99_Metadata/run_snapshot.json`
- a MATLAB-specific `matlab_resume_state.json` under the MATLAB output folder

The shared snapshot is used by the Python-side status surfaces and manifests.
The MATLAB resume file is used by the MATLAB wrapper to decide which batch to
reuse and which stage to run next.

These ledgers are not fully synchronized.

### 2. MATLAB resume is coarse

The MATLAB wrapper is restartable, but at stage granularity:

- `energy`
- `vertices`
- `edges`
- `network`

It determines the next stage from the last completed stage in the selected
batch. It does not currently advertise mid-stage restart semantics to the user.

This matters because the expensive `energy` stage can create many partial chunk
artifacts before the stage is complete. The presence of those artifacts does not
mean the next rerun will continue from the last written chunk.

### 3. Python resume is richer than MATLAB resume

The Python pipeline has a file-backed structured run state with:

- stage snapshots
- per-stage `resumed` flags
- `units_completed` and `units_total`
- stage manifests
- checkpoint files
- snapshot-based CLI status output

The MATLAB side does not currently expose comparable user-facing granularity in
comparison mode.

### 4. Failure detail is split across files

Today, failure evidence is distributed across:

- `matlab_run.log`
- `matlab_resume_state.json`
- the shared comparison snapshot
- the batch tree itself

The shared snapshot does not preserve enough of the MATLAB failure context to
let a user answer:

- what actually failed
- whether the run is stale vs still alive
- whether rerunning will reuse the current batch
- whether rerunning will restart the failed stage

## What The Current Evidence Suggests

Based on the current repo-local failures and the March 27, 2026 parity note, the
environmental `C:`-drive HDF5 failure is likely real. But even if that root
cause is fixed, the visibility problem remains:

- after a crash, the user still cannot tell what the rerun will do
- after a clean stop, the user still cannot tell which ledger to trust
- after a successful import-driven parity run, the user still cannot see why the
  Python stage restarts from `edges` instead of `energy`

So there are two related but distinct problems:

1. the operational failure itself
2. the lack of transparent rerun semantics

This plan addresses problem 2 directly.

## Questions That Need Answers Before Implementation

### Product and UX questions

1. What should be the single human-facing source of truth after implementation:
   `run_snapshot.json`, a manifest, a new MATLAB status file, or all three?
2. Is stage-level rerun visibility sufficient for MATLAB, or do we want chunk
   visibility for the `energy` stage as well?
3. When MATLAB dies mid-`energy`, should the status explicitly say
   "partial chunk artifacts exist but rerun will restart energy"?
4. How much of the MATLAB log should be persisted into metadata for failure
   review: the last 20 lines, 50 lines, or 100 lines?
5. Should low-free-space or OneDrive-backed output roots be surfaced as warnings
   in the run status and manifest?
6. Should the transparency surface live only in `slavv status`, or should the
   comparison entrypoint itself print a preflight resume decision before launch?

### Engineering questions

1. Can the current MATLAB wrapper compute and persist an authoritative "next
   action" summary before launching work, or should Python derive that summary by
   parsing the batch tree?
2. Do we want to detect stale `running` snapshots without PID tracking, using
   file mtime heuristics and log growth heuristics?
3. Should MATLAB state be mirrored into the shared snapshot as a dedicated
   `matlab_state` object, or should it be stored as optional task artifacts?
4. Is true mid-stage MATLAB energy resume desirable now, or should it be
   explicitly deferred as a separate higher-risk project?

### Scope questions

1. Is the first implementation limited to comparison-mode runs only?
2. Should the same status improvements also apply to standalone `run_matlab_cli`
   and `slavv import-matlab` workflows?
3. Should manifests and summaries be considered part of the authoritative
   transparency surface, or only derived reports?

## Recommended Implementation Direction

### Principle

Do not start by changing MATLAB computational behavior. First make rerun
semantics explicit and inspectable using the artifacts the repo already creates.

This reduces risk and immediately solves the user's black-box complaint.

### Phase 1: Make rerun decisions explicit

Goal: before launching MATLAB, determine and persist exactly what the next run
will do.

Proposed behavior:

- If no matching batch exists, mark the run as `fresh`.
- If a matching batch exists and the batch is complete, mark the run as
  `complete-noop`.
- If a matching batch exists and the last completed stage is `vertices`, mark
  the run as `resume-stage=edges`.
- If partial artifacts exist within a stage but the stage is not complete, mark
  the run as `restart-current-stage`, not `resume-in-stage`.

Proposed output fields:

- `matlab_batch_folder`
- `matlab_resume_mode`
- `matlab_last_completed_stage`
- `matlab_next_stage`
- `matlab_partial_stage_artifacts_present`
- `matlab_rerun_prediction`

Recommended code touch points:

- `workspace/scripts/cli/run_matlab_vectorization.m`
- `source/slavv/evaluation/comparison.py`
- `source/slavv/runtime/run_state.py`

### Phase 2: Mirror MATLAB state into the shared run snapshot

Goal: make `99_Metadata/run_snapshot.json` the place a human can inspect first.

Recommended changes:

- add a normalized MATLAB status payload to the shared snapshot
- update the `matlab_pipeline` optional task with richer artifacts
- store paths for:
  - `matlab_run.log`
  - `matlab_resume_state.json`
  - selected `batch_*` folder
- persist the predicted restart behavior into the shared snapshot before launch
- update the snapshot again after MATLAB returns

Recommended code touch points:

- `source/slavv/evaluation/comparison.py`
- `source/slavv/runtime/run_state.py`

### Phase 3: Improve failure reporting

Goal: if MATLAB fails, the shared metadata should explain the failure well
enough that the user does not need to manually dig through the batch tree.

Recommended changes:

- capture the tail of `matlab_run.log` on failure
- persist it in the shared snapshot and optionally in a dedicated
  `99_Metadata/matlab_failure_summary.json`
- record the MATLAB resume summary that was active at the time of failure
- distinguish:
  - `failed before batch selection`
  - `failed during energy`
  - `failed after stage boundary`
  - `stale-running-snapshot-suspected`

Recommended code touch points:

- `source/slavv/evaluation/comparison.py`
- `source/slavv/evaluation/management.py`
- `source/slavv/apps/cli.py`

### Phase 4: Upgrade CLI and manifest output

Goal: a user should be able to answer "what will rerun do?" by running one
status command or opening one manifest.

Recommended CLI additions:

- show MATLAB batch folder
- show MATLAB resume mode
- show last completed MATLAB stage
- show next MATLAB stage
- show explicit rerun prediction
- show whether Python will rerun from `energy`, `vertices`, `edges`, or
  `network`
- show the last failure summary if present

Recommended manifest additions:

- add a `Resume Semantics` section
- add an `Authoritative Files` section
- add a `Failure Summary` section when relevant

Recommended code touch points:

- `source/slavv/apps/cli.py`
- `source/slavv/runtime/run_state.py`
- `source/slavv/evaluation/management.py`

### Phase 5: Optional deeper MATLAB resume work

Goal: reduce wasted recomputation after a mid-`energy` crash.

This is explicitly a second project, not the first implementation target.

Potential direction:

- teach the MATLAB energy workflow to skip chunk outputs that already exist and
  pass integrity checks
- or persist a MATLAB-native chunk completion ledger that can safely drive true
  mid-stage resume

Risks:

- this code lives partly in `external/Vectorization-Public`
- HDF5 partial-write correctness must be validated carefully
- a naive "skip existing chunk file" rule could preserve corrupt or incomplete
  artifacts

Recommended code touch points if pursued:

- `external/Vectorization-Public/source/get_energy_V202.m`
- `workspace/scripts/cli/run_matlab_vectorization.m`

## Proposed File Changes

### New code

Recommended new module:

- `source/slavv/evaluation/matlab_status.py`

Suggested responsibilities:

- parse `matlab_resume_state.json`
- inspect batch completeness
- detect partial stage artifacts
- derive normalized rerun semantics
- tail `matlab_run.log`

### Existing files to change

- `source/slavv/evaluation/comparison.py`
- `source/slavv/runtime/run_state.py`
- `source/slavv/apps/cli.py`
- `source/slavv/evaluation/management.py`
- `workspace/scripts/cli/run_matlab_vectorization.m`

### Tests to add or extend

- `tests/unit/analysis/test_comparison_runtime.py`
- `tests/unit/runtime/test_run_state.py`
- new unit tests for normalized MATLAB status parsing and rerun prediction

### Docs to update after implementation

- `docs/COMPARISON_LAYOUT.md`
- possibly `docs/PARITY_FINDINGS_2026-03-27.md` with a short follow-up note if
  the transparency improvement changes the recommended workflow

## Acceptance Criteria

The implementation should be considered complete only if all of the following
are true:

1. After a crash, one status surface clearly says whether the next rerun is
   fresh, resumed, a no-op, or a stage restart.
2. The status surface names the exact `batch_*` folder being reused, if any.
3. The status surface distinguishes stage-level resume from mid-stage restart.
4. The shared comparison snapshot contains enough MATLAB-specific state that a
   user does not have to manually inspect `matlab_resume_state.json` first.
5. A failed MATLAB run stores a concise failure summary with log tail and next
   rerun prediction.
6. The manifest records the same rerun semantics as the status command.
7. Tests cover:
   - fresh run
   - completed batch no-op
   - stage-boundary resume
   - mid-stage partial-artifact restart
   - stale running snapshot
   - successful MATLAB import causing Python rerun from `edges`

## Recommended First Slice

The highest-value first slice is:

1. add a normalized MATLAB status parser
2. persist rerun prediction into `run_snapshot.json`
3. show that prediction in `slavv status`
4. capture log tail and failure summary on MATLAB failure

This delivers immediate operator clarity without changing computational
behavior, resume correctness, or the upstream MATLAB algorithm.

## Out of Scope For The First Pass

- fixing the `C:`-drive HDF5 write failure itself
- changing the MATLAB computational algorithm
- adding true chunk-level energy resume in upstream MATLAB code
- redesigning the directory layout

Those can be handled later, but they should not block the transparency fix.
