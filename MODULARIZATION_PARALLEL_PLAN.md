# Modularization Parallel Plan

## Purpose

This file turns the modularization roadmap into parallel workstreams that can be
picked up independently without creating unnecessary merge conflicts.

It is meant to answer five practical questions:

1. What architecture are we aiming for?
2. What can be worked on in parallel right now?
3. What workstreams must land before others?
4. Which files or packages belong to each workstream?
5. What validation is required before a workstream is considered done?

This plan complements the existing root-level `plan.md`, which is focused on
large-module facade breakdowns. This file is specifically about parallelizable
modularization work.

## Target Architecture

The desired package responsibilities are:

- `source/slavv/core/`
  - pure algorithms and stage logic
  - no direct ownership of resume policy, run snapshots, or CLI/UI concerns
- `source/slavv/runtime/`
  - run layout, snapshot persistence, resume guards, progress tracking, and
    event emission
- `source/slavv/io/`
  - import/export formats, MATLAB bridging, and serialization
- `source/slavv/apps/`
  - CLI and Streamlit presentation only
  - minimal orchestration logic
- `source/slavv/analysis/`
  - analysis and curation services built on stable typed results
- `source/slavv/visualization/`
  - visualization and curation UI layers built on stable typed results
- `source/slavv/parity/`
  - comparison workflows that compose `core`, `runtime`, and `io`
  - minimal duplication of orchestration logic

## Success Criteria

This effort is succeeding when:

- a bug in edge selection can be repaired without touching checkpoint logic
- a new export format can be added entirely inside `source/slavv/io/`
- Streamlit page modules mostly render UI and no longer own session
  orchestration
- CLI handlers mostly perform parsing and presentation, not business logic
- `core` stages exchange typed payloads instead of loosely-coupled dict
  contracts
- tests mirror owning package surfaces instead of historical task names

## Working Rules

- Keep public behavior stable unless a PR explicitly declares a contract change.
- Prefer add-wrap-migrate-delete over direct replacement.
- Keep public import paths stable by leaving thin facades in place while the
  internals move.
- Preserve MATLAB parity and legacy checkpoint compatibility unless a workstream
  explicitly updates those contracts and corresponding tests.
- Keep write scopes disjoint whenever possible so multiple branches can move in
  parallel.
- Finish each workstream with focused tests first, then repo-wide validation as
  appropriate.

## Validation Baseline

Every workstream should run the smallest relevant test surface first, then the
repo-standard commands from `AGENTS.md` as needed:

```powershell
python -m ruff format source tests
python -m ruff check source tests
python -m mypy
python -m pytest -m "unit or integration"
```

For UI-facing changes, include the relevant `dev/tests/ui/` coverage.
For parity-sensitive changes, include:

```powershell
python -m pytest dev/tests/diagnostic/test_comparison_setup.py
```

## Workstream Map

The workstreams below are ordered by dependency, but several can proceed in
parallel once their blockers are cleared.

### WS0 - Safety Net And Shared Builders

Goal:

- improve test ownership
- add shared fixture builders so refactors do not depend on production
  orchestration code just to create payloads

Primary write scope:

- `dev/tests/support/`
- `dev/tests/unit/`
- `dev/tests/integration/`
- `dev/tests/README.md`

Completed work:

- added `dev/tests/support/payload_builders.py`
- added `dev/tests/support/network_builders.py`
- added `dev/tests/support/run_state_builders.py`
- moved obvious ownership leaks such as `test_safe_unpickle.py`
- renamed task-history app tests to behavior names
- consolidated MAT/JSON visualizer export coverage under the visualization test surface
- split the large runtime file into focused runtime unit tests plus
  `dev/tests/integration/test_pipeline_run_state_integration.py`
- adopted shared builders in several app, io, runtime, visualization, and parity tests

Reference:

- `WS0_SAFETY_NET_CHECKLIST.md` records the completed safety-net reshaping work

Can run in parallel with:

- follow-on workstreams only; treat WS0 as the completed safety-net baseline

Must land before:

- already satisfied for active work

### WS1 - Typed Result Models

Goal:

- replace implicit dict contracts with typed payloads while preserving existing
  compatibility surfaces

Primary write scope:

- `source/slavv/models/` or `source/slavv/core/types.py`
- `source/slavv/core/`
- small compatibility adapters in `analysis`, `visualization`, and `apps`

Candidate files:

- `source/slavv/core/pipeline.py`
- `source/slavv/core/_vertices/extraction.py`
- `source/slavv/core/_edges/standard.py`
- `source/slavv/analysis/automatic_curator.py`
- `source/slavv/analysis/ml_curator.py`
- `source/slavv/visualization/interactive_curator.py`

Deliverables:

- typed models such as `EnergyResult`, `VertexSet`, `EdgeSet`, `NetworkResult`
- temporary `from_dict()` and `to_dict()` compatibility helpers
- tests for payload invariants and adapter behavior

Completed so far:

- added `source/slavv/models/`
- added dict compatibility adapters for the initial result models
- added a shared `normalize_pipeline_result()` helper so adapter logic has a
  single repair point
- added focused adapter tests
- routed `SLAVVProcessor.process_image()` final and early-stop returns through
  the typed compatibility seam while preserving dict-shaped public results
- adopted typed result normalization in app/report helpers such as
  `share_report.py`, `curation_state.py`, and `web_app_artifacts.py`
- added `apps/dashboard_state.py` so dashboard context loading and stat
  resolution are normalized and directly unit-tested outside the Streamlit
  page module
- added `apps/analysis_state.py` so analysis-page payload handling and fallback
  stats resolution are normalized and directly unit-tested outside the
  Streamlit page module
- added `apps/visualization_state.py` so visualization-page payload handling is
  normalized and directly unit-tested outside the Streamlit page module
- adopted typed result normalization in visualization statistics/export helpers
  under `network_plot_statistics.py`
- adopted typed result normalization in `analysis/ml_curator.py` training-data
  preparation

Still to do:

- continue shrinking direct dict access in additional app, visualization, and
  analysis entrypoints
- decide whether the next WS1 step should cover more consumers or introduce
  additional typed component models for deeper internal stages

Can run in parallel with:

- WS3 after the basic model package lands

Must land before:

- WS2, WS4, WS5, WS6, WS7

### WS2 - Pipeline Workflow Extraction

Goal:

- separate pipeline sequencing and run orchestration from `SLAVVProcessor`

Primary write scope:

- `source/slavv/workflows/`
- `source/slavv/core/pipeline.py`

Candidate files:

- `source/slavv/core/pipeline.py`

Deliverables:

- `source/slavv/workflows/pipeline_runner.py`
- `source/slavv/workflows/pipeline_stages.py`
- a thinner `SLAVVProcessor` facade
- focused tests for the runner and stage sequencing

Completed so far:

- added `source/slavv/workflows/`
- moved pipeline result finalization and stop-after handling into
  `workflows/pipeline_results.py`
- moved pipeline setup helpers such as stage validation, run-context creation,
  rerun-flag calculation, progress emission, and preprocessing into
  `workflows/pipeline_setup.py`
- moved shared result-bundle/progress/stop-after advancement into
  `workflows/pipeline_runner.py`
- moved the sequential stage loop itself into
  `workflows/pipeline_runner.py`
- moved shared stage-resolution control flow into
  `workflows/pipeline_stages.py`
  including controller lookup and failure tracking
- moved repeated stage checkpoint load/save/complete logic into
  `workflows/stage_checkpoints.py`
- added focused workflow helper tests while preserving `SLAVVProcessor`
  behavior

Still to do:

- extract stage sequencing and stage-resolution orchestration into dedicated
  workflow modules
- keep thinning `core/pipeline.py` until it mainly delegates orchestration

Can run in parallel with:

- WS3 once WS1 is stable

Must land before:

- WS4
- parts of WS5
- parts of WS6

### WS3 - Run-State Service Split

Goal:

- split `RunContext` into smaller runtime services without changing the external
  run-state API

Primary write scope:

- `source/slavv/runtime/_run_state/`
- `source/slavv/runtime/run_state.py`

Candidate files:

- `source/slavv/runtime/_run_state/context.py`
- `source/slavv/runtime/_run_state/io.py`
- `source/slavv/runtime/_run_state/models.py`
- `source/slavv/runtime/_run_state/status.py`
- `source/slavv/runtime/run_state.py`

Deliverables:

- `layout.py`
- `snapshot_store.py`
- `resume_guard.py`
- `progress.py`
- a compatibility `RunContext` facade

Can run in parallel with:

- WS2 after WS1

Must land before:

- parity cleanup in WS8
- app/service cleanup in WS5 where run-task mutation is involved

### WS4 - Edge Workflow Decomposition

Goal:

- break the edge pipeline into generation, selection, and persistence/audit
  modules with one coordinator

Primary write scope:

- `source/slavv/core/_edges/`
- `source/slavv/core/_edge_candidates/`
- `source/slavv/core/_edge_selection/`
- `source/slavv/core/edges.py`
- `source/slavv/core/edge_candidates.py`
- `source/slavv/core/edge_selection.py`

Candidate files:

- `source/slavv/core/_edges/resumable.py`
- `source/slavv/core/_edge_candidates/generate.py`
- `source/slavv/core/_edge_selection/workflow.py`

Deliverables:

- `edge_generation/` or equivalent focused internal modules
- `edge_selection/` cleanup-only logic
- `edge_resume/` persistence and audit writers
- `EdgeWorkflow` coordinator
- tests reorganized to mirror internal seams

Can run in parallel with:

- WS5 if write scopes stay disjoint
- WS6 if write scopes stay disjoint

Must land before:

- any attempt to narrow `core` public API

### WS5 - App Service Extraction

Goal:

- remove orchestration from Streamlit pages and replace the `web_app` facade
  dependency pattern with explicit services

Primary write scope:

- `source/slavv/apps/`

Candidate files:

- `source/slavv/apps/web_app.py`
- `source/slavv/apps/web_app_processing_page.py`
- `source/slavv/apps/web_app_curation_page.py`
- `source/slavv/apps/web_app_visualization_page.py`
- `source/slavv/apps/web_app_dashboard_page.py`
- `source/slavv/apps/web_app_artifacts.py`
- `source/slavv/apps/curation_state.py`

Deliverables:

- `app_state.py`
- `processing_service.py`
- `dashboard_service.py`
- `export_service.py`
- page modules that mostly render UI

Can run in parallel with:

- WS4
- WS6

Must land before:

- any final cleanup that removes facade compatibility from `web_app.py`

### WS6 - CLI Service Extraction

Goal:

- separate CLI request handling from console presentation

Primary write scope:

- `source/slavv/apps/`

Candidate files:

- `source/slavv/apps/cli.py`
- `source/slavv/apps/cli_commands.py`
- `source/slavv/apps/cli_parser.py`
- `source/slavv/apps/cli_shared.py`

Deliverables:

- command service modules returning structured results
- a thinner `cli.py` entrypoint
- less reliance on barrel re-exports of underscore helpers

Can run in parallel with:

- WS5
- WS4

Must land before:

- any final CLI public API narrowing

### WS7 - IO Ownership Consolidation

Goal:

- make `io` the single owner of formats and MATLAB adaptation

Primary write scope:

- `source/slavv/io/`
- small wrapper updates in `visualization` and `apps`

Candidate files:

- `source/slavv/io/network_io.py`
- `source/slavv/io/matlab_bridge.py`
- `source/slavv/visualization/network_plots.py`

Deliverables:

- `source/slavv/io/models.py`
- `source/slavv/io/formats/json.py`
- `source/slavv/io/formats/csv.py`
- `source/slavv/io/formats/casx.py`
- `source/slavv/io/formats/vmv.py`
- `source/slavv/io/formats/mat.py`
- split MATLAB reader, adapter, mapper, and import service modules

Can run in parallel with:

- WS5
- WS6

Must land before:

- public API cleanup for visualization export wrappers

### WS8 - Parity Orchestration Consolidation

Goal:

- finish reducing duplicated comparison orchestration by making
  `parity/comparison.py` a thin compatibility facade over the internal package

Primary write scope:

- `source/slavv/parity/`

Candidate files:

- `source/slavv/parity/comparison.py`
- `source/slavv/parity/_comparison/`
- `source/slavv/parity/workflow_assessment.py`
- `source/slavv/parity/reporting.py`

Deliverables:

- one authoritative comparison orchestration path
- thinner parity facades
- reduced duplication across comparison helpers

Can run in parallel with:

- WS7 after runtime and pipeline seams stabilize

Must land before:

- final public API cleanup

### WS9 - Public API Cleanup

Goal:

- remove accidental compatibility surfaces after the new modular boundaries are
  in place and tested

Primary write scope:

- `source/slavv/__init__.py`
- `source/slavv/apps/cli.py`
- `source/slavv/core/edge_candidates.py`
- any remaining facade modules exporting private helpers

Deliverables:

- explicitly documented supported public API
- reduced barrel re-exports
- private helpers no longer treated as package API

Can run in parallel with:

- none; this is the final cleanup wave

Must land after:

- WS4, WS5, WS6, WS7, WS8

## Dependency Summary

In short form:

- WS0 unlocks everything.
- WS1 unlocks WS2 and makes the later layers cleaner.
- WS2 and WS3 can proceed in parallel after WS1 stabilizes.
- WS4 depends mostly on WS2.
- WS5 and WS6 can proceed in parallel once WS2 and WS3 are in decent shape.
- WS7 can proceed once the app and CLI surfaces are no longer tightly coupled
  to old wrappers.
- WS8 should wait until the workflow and runtime seams are stable.
- WS9 is the cleanup pass after the new architecture is already in use.

## Suggested Parallel Execution Waves

### Wave A

- WS0 - Safety Net And Shared Builders
- WS1 - Typed Result Models

These are the least glamorous but highest-leverage unlocks.

### Wave B

- WS2 - Pipeline Workflow Extraction
- WS3 - Run-State Service Split

These can run side by side if one branch owns `core/workflows` and the other
owns `runtime/_run_state`.

### Wave C

- WS4 - Edge Workflow Decomposition
- WS5 - App Service Extraction
- WS6 - CLI Service Extraction

These are parallel-friendly if teams avoid overlapping facade cleanup in the
same PR.

### Wave D

- WS7 - IO Ownership Consolidation
- WS8 - Parity Orchestration Consolidation

These should start after the workflow and runtime seams are more stable.

### Wave E

- WS9 - Public API Cleanup

This is the final tightening pass after the rest has landed.

## Non-Overlapping PR Guidance

To reduce conflicts, prefer PRs with these write boundaries:

- one PR changes only `dev/tests/support` plus a small set of moved tests
- one PR changes only `source/slavv/models` plus adapter call sites
- one PR changes only `source/slavv/workflows` plus the `core/pipeline.py`
  facade
- one PR changes only `source/slavv/runtime/_run_state`
- one PR changes only `source/slavv/core/_edge*` internals
- one PR changes only Streamlit pages and app services
- one PR changes only CLI services
- one PR changes only `source/slavv/io`
- one PR changes only `source/slavv/parity/_comparison` and comparison facades

Avoid combining facade cleanup and deep internal movement in the same PR unless
the surface is truly tiny.

## Per-Workstream Done Criteria

Each workstream is done when:

- the new module boundary exists and is tested
- old import paths still work unless the PR explicitly removes them
- focused tests for the owning area pass
- repo-standard lint and type checks pass for the touched surface
- the original hotspot module is smaller or thinner than before
- the follow-on workstream no longer needs to duplicate the extracted logic

## First Recommended PR Sequence

If this work starts immediately, the best first sequence is:

1. WS0 - Safety Net And Shared Builders
2. WS1 - Typed Result Models
3. WS2 - Pipeline Workflow Extraction
4. WS3 - Run-State Service Split

That sequence gives the biggest maintainability gain while keeping parity risk
manageable.

## Notes For Parallel Contributors

- Read `AGENTS.md`, `docs/README.md`, and `dev/tests/README.md` before starting.
- If your work touches parity-sensitive code, check `TODO.md` and the current
  canonical run roots before changing workflow assumptions.
- If your work touches checkpoint or staged comparison layout logic, preserve
  compatibility with:
  - `source/slavv/parity/run_layout.py`
  - `source/slavv/runtime/run_state.py`
- Do not narrow a public surface in the same PR that introduces the replacement
  surface unless the call sites are fully migrated and covered by tests.
