# Design Document: Comparison Layout Smoothing

## Overview

This design introduces a lightweight organization layer above existing run roots
while preserving the established staged layout used by parity workflows.

Historical note:

- This archived design captures the migration plan and the repository scan at
  the time it was completed. Example run roots such as
  `20260413_release_verify` are historical migration examples, not the current
  active parity evidence roots.

Current staged layout remains unchanged inside each run root:

- `01_Input/`
- `02_Output/`
- `03_Analysis/`
- `99_Metadata/`

New design elements:

1. experiment grouping under `slavv_comparisons/experiments/`
2. global pointer files under `slavv_comparisons/pointers/`
3. per-run lifecycle metadata in `99_Metadata/status.json`
4. per-experiment machine-readable index files

Repository-informed constraint:

- some current roots are direct staged runs
- some current roots are aggregate containers with `run_*` children

The migration design must normalize both shapes without data loss.

## Design Principles

- Minimal disruption: do not alter run-internal staged contracts.
- Backward compatibility: allow legacy top-level run roots during migration.
- Explicit references: make canonical paths discoverable without guesswork.
- Safe cleanup: require lifecycle metadata and supersession evidence.
- Script-first execution: prefer dedicated maintenance scripts over long ad hoc
  shell commands.
- Auditability: every migration step should be reconstructable from report files.

## Target Layout

```text
slavv_comparisons/
  experiments/
    <experiment_slug>/
      index.json
      runs/
        <YYYYMMDD_HHMMSS_label>/
          01_Input/
          02_Output/
          03_Analysis/
          99_Metadata/
            status.json
  pointers/
    latest_completed.txt
    canonical_acceptance.txt
    best_saved_batch.txt
```

## Components

### 1. Experiment Grouping

- Group related runs by intent (`release-verify`, `saved-batch`, `consistency`, etc.).
- Keep run roots immutable once created (except status metadata updates).
- Generate `index.json` per experiment as a compact summary table.

Initial slug mapping based on the historical repository scan:

- `20260327_150656_clean_parity` -> `saved-batch`
- `20260327_161610_clean_python_full` -> `python-full`
- `20260330_parity_full_postfix` -> `postfix-parity`
- `20260330_cross_compare_postfix` -> `postfix-cross-compare`
- `20260328_023500_matlab_consistency` -> `matlab-consistency`
- `20260328_142659_python_consistency` -> `python-consistency`
- `20260330_python_consistency_postfix` -> `python-consistency`
- `20260401_live_parity_retry` -> `live-parity`
- `20260413_release_verify` -> `release-verify`

Suggested `index.json` entry schema:

```json
{
  "run_path": "experiments/release-verify/runs/20260413_144432_live_canonical",
  "timestamp": "2026-04-13T14:44:32",
  "state": "completed",
  "quality_gate": "partial",
  "parity": {
    "vertices": "pass",
    "edges": "fail",
    "strands": "fail"
  }
}
```

### 2. Pointer Files

- Store exactly one repo-relative run path per file.
- Keep pointers human-editable and script-friendly.
- Use pointers in docs to reduce stale hardcoded paths.

Historical example pointer file content:

```text
experiments/release-verify/runs/20260413_144432_live_canonical
```

### 3. Run Lifecycle Metadata

Add `99_Metadata/status.json` in each managed run root.

Proposed schema:

```json
{
  "state": "completed",
  "retention": "keep",
  "quality_gate": "partial",
  "supersedes": null,
  "superseded_by": null,
  "notes": "Canonical release verification run"
}
```

State transitions:

- `incomplete -> completed`
- `incomplete -> failed`
- `failed -> superseded` (when a replacement run is validated)
- any state -> `archived` (manual policy decision)

Status inference precedence:

1. explicit `99_Metadata/status.json` (if present)
2. `99_Metadata/run_snapshot.json` status field (direct roots)
3. analysis artifacts presence (`03_Analysis/comparison_report.json`,
   `03_Analysis/summary.txt`)
4. aggregate-root child rollup from `run_*` directories
5. fallback `incomplete`

Aggregate-root policy:

- store optional aggregate `status.json` at the root
- store child-level status for each `run_*` when child metadata exists
- compute aggregate state as:
  - `completed` if all children are completed
  - `failed` if any child failed and no replacement exists
  - `incomplete` otherwise

### 4. Consolidation Workflow

1. Discover run roots and collect key artifacts.
2. Detect root shape (direct staged vs aggregate root with `run_*` children).
3. Normalize date-first naming where needed.
4. Move/group runs under `experiments/<slug>/runs/`.
5. Write or update `99_Metadata/status.json`.
6. Rebuild experiment `index.json` files.
7. Update pointer files.
8. Remove empty folders.

Implementation note:

- execute workflow through maintained scripts in
  `dev/scripts/maintenance/`
- persist a migration report under `99_Metadata/` or a dedicated
  maintenance report path

## Migration Strategy

Phase A (non-destructive):

- detect and report candidate grouping/renames
- generate proposed pointer/index outputs
- no deletions

Phase B (apply):

- move/rename runs with conflict checks
- write status metadata and indexes
- update pointers
- prune empty directories

Phase C (doc sync):

- update chapter/reference links to pointer files or final paths
- verify all cited paths exist

## Failure Handling

- If required metadata is missing, mark `state=incomplete` and continue.
- If move target exists, skip with warning and record conflict.
- Never delete non-empty run roots automatically unless explicitly approved
  through an allow-list.
- Run doc-link verification after move/rename operations and block finalization
  when stale references remain unresolved.

## Known Weakness Mitigations

This plan explicitly mitigates known execution weak spots observed in practice:

- Long PowerShell one-liners are brittle in this environment.
  - Mitigation: script-first implementation and short command invocations.
- Path renames can leave stale documentation references.
  - Mitigation: mandatory post-migration grep/report gate before completion.
- Aggressive cleanup can remove useful evidence.
  - Mitigation: explicit allow-list for non-empty deletion and retention flags.
- Metadata can be sparse or inconsistent across run families.
  - Mitigation: deterministic status inference precedence with aggregate rollups.

## Compatibility Notes

- Existing CLI behavior remains unchanged.
- Existing staged artifact readers continue to work inside each run root.
- This design is filesystem-organization only; no parity algorithm changes.
