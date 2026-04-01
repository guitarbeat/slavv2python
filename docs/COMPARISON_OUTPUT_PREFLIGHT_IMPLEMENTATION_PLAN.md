# Comparison Output Preflight Implementation Plan

Status: Draft
Date: 2026-04-01

## Objective

Prevent obviously risky or doomed MATLAB/comparison runs before they start by
adding an enforced output-root preflight that answers these questions up front:

1. Is the selected output root writable?
2. Does the target drive have enough free space for a realistic MATLAB run?
3. Is the path in a risky location such as a OneDrive-synced folder?
4. Will the run fail fast with a clear explanation, or proceed with an explicit
   warning?
5. Where should a user look after preflight to confirm the decision that was
   made?

This plan focuses on launch safety and operator clarity. It does not try to fix
the underlying MATLAB HDF5 behavior itself.

## Why This Matters

The repo already contains clear evidence that output-root selection is an
operational weakness:

- [docs/PARITY_FINDINGS_2026-03-27.md](PARITY_FINDINGS_2026-03-27.md) records
  repeated fresh MATLAB failures on `C:` during `energy`, while the same work
  succeeded immediately on `D:` with ample free space.
- [workspace/reports/matlab_standalone_consistency_2026-03-28.md](../workspace/reports/matlab_standalone_consistency_2026-03-28.md)
  shows healthy standalone MATLAB runs taking roughly `743s` to `909s`, with
  `energy` dominating runtime.
- [source/slavv/evaluation/setup_checks.py](../source/slavv/evaluation/setup_checks.py)
  already contains a disk-space check, which means the repo recognizes the
  problem, but that logic is not currently driving the real comparison launch
  path.

The practical cost is high:

- MATLAB `energy` failures happen late enough to waste substantial time.
- Comparison-mode runs write large staged artifacts under the chosen root.
- The current workflow lets a risky output root survive until MATLAB fails at
  runtime instead of rejecting or warning at launch.

## Files Already Available

### Existing evidence

- `docs/PARITY_FINDINGS_2026-03-27.md`
- `workspace/reports/matlab_standalone_consistency_2026-03-28.md`
- `workspace/reports/python_matlab_parity_postfix_2026-03-30.md`
- `docs/COMPARISON_RESUME_TRANSPARENCY_IMPLEMENTATION_PLAN.md`

### Existing code paths involved

- `source/slavv/evaluation/comparison.py`
- `source/slavv/evaluation/setup_checks.py`
- `source/slavv/evaluation/management.py`
- `source/slavv/apps/cli.py`
- `workspace/scripts/cli/compare_matlab_python.py`
- `workspace/scripts/cli/run_matlab_cli.bat`
- `workspace/scripts/cli/run_matlab_cli.sh`

### Existing tests and diagnostics

- `tests/diagnostic/test_comparison_setup.py`
- `tests/unit/analysis/test_comparison_runtime.py`
- `tests/unit/runtime/test_run_state.py`

## Current Behavior

### 1. Preflight logic exists but is not authoritative

`setup_checks.py` has a `Validator.check_disk_space(...)` helper, but the live
comparison path in `orchestrate_comparison(...)` does not appear to invoke it
before launching MATLAB.

That means the repo has validation code, but not a single authoritative
preflight gate.

### 2. The real launch path is permissive

`run_matlab_vectorization(...)` in
`source/slavv/evaluation/comparison.py` validates file existence and then
launches MATLAB. It creates the output directory eagerly and starts the run
without checking:

- free space
- writability beyond directory creation
- suspicious sync-backed path roots
- whether the output root is likely to be slow or unstable for HDF5-heavy work

### 3. The Windows and POSIX launchers are thin wrappers

`workspace/scripts/cli/run_matlab_cli.bat` and `.sh` create the output
directory and log file, then call MATLAB. They do not currently perform a
path-health or storage-health preflight.

### 4. Diagnostics are detached from real orchestration

`tests/diagnostic/test_comparison_setup.py` checks for files and default paths,
but it does not validate the actual caller-selected output root used by a live
comparison run.

### 5. Failures are discovered too late

Today the user typically learns the chosen output root was a bad idea only
after MATLAB has already spent time inside `energy` and returned a nonzero
exit.

## What The Current Evidence Suggests

The March 27 and March 28 evidence points to a repo deficiency that is broader
than a single machine-specific failure:

1. MATLAB comparison runs are sensitive to output-root conditions.
2. The repo already knows enough to detect at least some bad conditions
   earlier.
3. The current orchestration does not convert that knowledge into a consistent
   preflight decision.

This is a real implementation gap, not just a documentation gap.

## Questions That Need Answers Before Implementation

### Product and UX questions

1. Which preflight findings should be fatal versus warning-only?
2. Should OneDrive-backed paths always warn, or only when used for MATLAB
   outputs?
3. Should the CLI offer an explicit override such as
   `--allow-risky-output-root`?
4. Should preflight findings be shown only before launch, or also in
   `slavv status` and the manifest afterward?

### Engineering questions

1. Should the minimum free-space threshold be fixed, configurable, or estimated
   from the input volume size?
2. How should Windows-specific path heuristics be implemented:
   path substring checks, environment-variable comparisons, or both?
3. Should launcher scripts perform their own minimal checks, or should Python be
   the single source of truth and pass a rendered decision downstream?
4. Should preflight results be stored in `run_snapshot.json`, a dedicated JSON
   artifact, or both?

### Scope questions

1. Is the first pass limited to comparison-mode launches?
2. Should standalone MATLAB launcher usage benefit from the same safeguards?
3. Should native Python-only runs also receive the same output-root preflight,
   even if the immediate pain point is on the MATLAB side?

## Recommended Implementation Direction

### Principle

Treat output-root health as first-class runtime state, not as an optional
manual checklist.

If a run is risky enough that an experienced operator would hesitate to launch
it, the software should say so before MATLAB starts.

### Phase 1: Introduce a normalized preflight report

Goal: compute one structured preflight decision for the selected output root.

Recommended checks:

- target path resolves successfully
- parent directory exists or can be created
- output root is writable
- free space exceeds a configured minimum threshold
- path is on a potentially risky sync-backed root such as OneDrive
- path is local versus obviously remote/network-backed when detectable

Proposed output fields:

- `output_root`
- `resolved_output_root`
- `preflight_status`
- `free_space_gb`
- `required_space_gb`
- `writable`
- `onedrive_suspected`
- `warnings`
- `errors`
- `recommended_action`

Recommended code touch points:

- `source/slavv/evaluation/setup_checks.py`
- new module: `source/slavv/evaluation/preflight.py`

### Phase 2: Make preflight authoritative in comparison orchestration

Goal: no comparison-mode MATLAB launch should occur without a recorded preflight
decision.

Recommended changes:

- run preflight before `run_matlab_vectorization(...)`
- fail fast on fatal findings
- allow warning-only findings to proceed with an explicit console message
- persist the preflight decision into run metadata before launch

Recommended code touch points:

- `source/slavv/evaluation/comparison.py`
- `workspace/scripts/cli/compare_matlab_python.py`

### Phase 3: Persist and surface the decision clearly

Goal: after the run starts or fails, the operator should still be able to see
why the output root was accepted or rejected.

Recommended changes:

- add a dedicated metadata artifact such as
  `99_Metadata/output_preflight.json`
- mirror the high-level decision into `run_snapshot.json`
- add a `Preflight` section to the manifest
- surface warnings in `slavv status` when relevant

Recommended code touch points:

- `source/slavv/runtime/run_state.py`
- `source/slavv/evaluation/management.py`
- `source/slavv/apps/cli.py`

### Phase 4: Add launcher-level safety net

Goal: direct launcher usage should not completely bypass the improvement.

Recommended changes:

- keep Python as the authoritative preflight engine
- optionally add minimal shell-level checks in `run_matlab_cli.bat` and `.sh`
  for obviously invalid conditions such as missing parent directories or
  immediate log-file creation failures
- keep the shell logic thin so behavior does not diverge from Python

Recommended code touch points:

- `workspace/scripts/cli/run_matlab_cli.bat`
- `workspace/scripts/cli/run_matlab_cli.sh`

### Phase 5: Expand diagnostics and policy controls

Goal: make the preflight behavior tunable without making it ambiguous.

Possible additions:

- configurable free-space threshold
- explicit override flag for warning-level findings
- stricter default policy for MATLAB-enabled comparison runs than for
  Python-only runs

## Proposed File Changes

### New code

Recommended new module:

- `source/slavv/evaluation/preflight.py`

Suggested responsibilities:

- compute output-root health
- normalize warnings versus fatal errors
- serialize preflight reports
- expose reusable helpers for CLI and orchestration code

### Existing files to change

- `source/slavv/evaluation/comparison.py`
- `source/slavv/evaluation/setup_checks.py`
- `source/slavv/runtime/run_state.py`
- `source/slavv/evaluation/management.py`
- `source/slavv/apps/cli.py`
- `workspace/scripts/cli/compare_matlab_python.py`
- `workspace/scripts/cli/run_matlab_cli.bat`
- `workspace/scripts/cli/run_matlab_cli.sh`

### Tests to add or extend

- `tests/unit/analysis/test_comparison_runtime.py`
- `tests/unit/runtime/test_run_state.py`
- new unit tests for output-root preflight classification and serialization
- extend diagnostics to verify that risky output roots are reported clearly

### Docs to update after implementation

- `docs/COMPARISON_LAYOUT.md`
- `docs/README.md`
- possibly `docs/PARITY_FINDINGS_2026-03-27.md` with a short note pointing to
  the new safer workflow

## Acceptance Criteria

The implementation should be considered complete only if all of the following
are true:

1. A comparison run records an output-root preflight result before MATLAB
   launches.
2. Low-free-space conditions are detected and surfaced before launch.
3. Suspicious sync-backed output roots such as OneDrive paths are surfaced as
   warnings before launch.
4. Fatal preflight failures prevent MATLAB from starting and explain why.
5. Warning-level findings persist into the manifest and shared run snapshot.
6. `slavv status` or an equivalent status surface shows the recorded preflight
   outcome.
7. Tests cover:
   - healthy local output root
   - low-free-space root
   - unwritable root
   - OneDrive-suspected path
   - preflight artifact persistence on both success and blocked launch

## Recommended First Slice

The highest-value first slice is:

1. add a normalized output-root preflight helper
2. invoke it from `orchestrate_comparison(...)` before MATLAB launch
3. persist the result into `99_Metadata/output_preflight.json`
4. block obviously fatal launches and print clear warnings for risky-but-allowed
   paths

This delivers immediate protection against the documented `C:`/path-selection
failure mode without changing MATLAB algorithms or comparison semantics.

## Out of Scope For The First Pass

- fixing the underlying MATLAB HDF5 behavior itself
- automatically relocating runs onto a different drive
- estimating exact storage needs for every dataset size
- redesigning the staged comparison directory layout
- adding true resume semantics for partially failed MATLAB stages

Those are valuable follow-on efforts, but they should not block the preflight
guardrail.
