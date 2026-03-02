# Experiments Directory Review

## Scope Reviewed

I reviewed `workspace/experiments` and mapped its directory structure.

## What I Learned

1. The top-level folder mixes two organization styles:
   - Date hierarchy style: `2026/02-February`
   - Timestamped run folders: `YYYYMMDD_HHMMSS_<run_type>`
2. Current run folders found:
   - `20260206_173559_matlab_run`
   - `20260209_173027_full_run`
   - `20260209_173134_full_run`
   - `20260209_173550_full_run`
   - `20260210_100526_full_run`
   - `20260210_101213_manual_run`
3. Most runs follow a 3-stage pipeline layout:
   - `01_Input`
   - `02_Output`
   - `03_Analysis`
4. `matlab_results` appears under `01_Input` in multiple runs, which implies MATLAB output is treated as upstream pipeline input to Python steps.
5. `python_results/checkpoints` appears under `02_Output` where present, which implies checkpointed Python processing outputs.
6. `20260206_173559_matlab_run` is an outlier because it also has duplicated result roots at run top-level:
   - `matlab_results`
   - `python_results`
   This duplicates content also represented in stage subfolders.
7. `20260210_101213_manual_run` contains multiple MATLAB batch folders (`batch_*`), showing repeated manual executions grouped inside one run container.
8. In this workspace snapshot, visible file count under `workspace/experiments` is `0`; only directory scaffolding is visible. This suggests file payloads may be excluded, not synced locally, or generated elsewhere.

## Structural Issues

1. Mixed hierarchy conventions (calendar-based + run-based) increase cognitive load.
2. Potential duplication of results paths (root-level + stage-level) risks drift and ambiguity.
3. Naming is mostly strong, but placement is not fully normalized across runs.
4. Empty directory-heavy trees make it hard to determine data freshness and reproducibility state.

## Recommended Refactor

1. Pick one top-level convention and enforce it:
   - Recommended: keep only run folders at top-level (`YYYYMMDD_HHMMSS_<type>`).
2. Normalize run schema for every run:
   - `01_Input/`
   - `02_Output/`
   - `03_Analysis/`
   - Optional: `99_Metadata/` for manifests and run summaries.
3. Eliminate duplicate top-level `matlab_results`/`python_results` inside run roots.
4. Add one required run manifest per run (for example `99_Metadata/run_manifest.md`) with:
   - run ID, type, timestamps
   - source dataset
   - MATLAB batch IDs
   - expected outputs
   - completion status
5. Add retention policy notes:
   - whether data files are intentionally excluded from git
   - where full payloads live (artifact storage, network drive, etc.)

## Implemented Refactor

Applied in-place normalization to `workspace/experiments`:

1. Flattened top-level organization to run folders only (removed dated hierarchy branch).
2. Standardized each run to:
   - `01_Input/`
   - `02_Output/`
   - `03_Analysis/`
   - `99_Metadata/`
3. Moved duplicate root artifacts to staged locations where applicable.
4. Added `99_Metadata/run_manifest.md` for each run.
5. Added `.gitkeep` placeholders in stage folders so structure is stable and trackable.

## Comparative Analysis Improvements Implemented

Code updates were made to make comparative analysis resilient to both layouts:

1. Added run-layout resolution in evaluation management tooling.
2. Updated run discovery to resolve staged roots correctly and avoid stage-folder false positives.
3. Updated summary generation to read report/status from staged paths.
4. Updated comparison orchestration to write:
   - MATLAB output to `01_Input/matlab_results`
   - Python output to `02_Output/python_results`
   - reports/summaries to `03_Analysis`
   - manifests to `99_Metadata/run_manifest.md`
5. Added focused tests for staged layout behavior:
   - `tests/unit/analysis/test_evaluation_layouts.py`
