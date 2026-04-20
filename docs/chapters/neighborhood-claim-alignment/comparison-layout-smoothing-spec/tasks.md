# Implementation Plan: Comparison Layout Smoothing

## Overview

This plan implements the new comparison layout organization model with minimal
risk and full backward compatibility for existing staged runs.

Historical note:

- The repository scan examples in this archived plan reflect the migration
  moment when the layout work was completed. Names such as
  `20260413_release_verify` are historical inventory examples, not the current
  active evidence roots for parity work.

Repository scan grounding for this plan:

- direct staged roots: `20260327_150656_clean_parity`,
  `20260327_161610_clean_python_full`, `20260330_parity_full_postfix`,
  `20260330_cross_compare_postfix`, `20260401_live_parity_retry`,
  `20260413_release_verify`
- aggregate roots with `run_*` children: `20260328_023500_matlab_consistency`,
  `20260328_142659_python_consistency`, `20260330_python_consistency_postfix`

## Phase 1: Spec and Schema Finalization

- [x] 1. Confirm run lifecycle schema
  - [x] 1.1 Finalize `99_Metadata/status.json` required fields and enums
  - [x] 1.2 Document state transition rules
  - [x] 1.3 Finalize aggregate-root rollup policy for `run_*` containers

- [x] 2. Confirm experiment slug taxonomy
  - [x] 2.1 Define initial slug set (`release-verify`, `saved-batch`, `python-full`, `postfix-parity`, `postfix-cross-compare`, `python-consistency`, `matlab-consistency`, `live-parity`)
  - [x] 2.2 Document slug assignment rules for new runs
  - [x] 2.3 Freeze initial mapping for currently retained run roots

## Phase 2: Filesystem Scaffolding

- [x] 3. Create root scaffolding
  - [x] 3.1 Create `slavv_comparisons/experiments/`
  - [x] 3.2 Create `slavv_comparisons/pointers/`

- [x] 4. Seed pointer files
  - [x] 4.1 Create `latest_completed.txt`
  - [x] 4.2 Create `canonical_acceptance.txt`
  - [x] 4.3 Create `best_saved_batch.txt`

## Phase 3: Non-Destructive Inventory

- [x] 5. Implement inventory script/report in `dev/scripts/maintenance/`
  - [x] 5.1 Detect current run roots and staged completeness
  - [x] 5.2 Extract run status from `run_snapshot.json` and analysis artifacts
  - [x] 5.3 Detect root shape (`direct` vs `aggregate`)
  - [x] 5.4 Propose date-first renames where needed
  - [x] 5.5 Propose experiment slug assignment per run
  - [x] 5.6 Emit machine-readable dry-run report (moves, skips, conflicts, delete candidates)

- [x] 6. Validate inventory output manually
  - [x] 6.1 Confirm canonical runs are marked `retention=keep`
  - [x] 6.2 Confirm failed runs have replacement linkage before cleanup eligibility
  - [x] 6.3 Confirm aggregate roots have consistent rollup status behavior

## Phase 4: Apply Consolidation

- [x] 7. Apply grouping and naming updates
  - [x] 7.1 Move runs into `experiments/<slug>/runs/`
  - [x] 7.2 Apply date-first rename normalization
  - [x] 7.3 Preserve run-internal staged folder structure
  - [x] 7.4 Support idempotent re-run after partial migration

- [x] 8. Write status metadata
  - [x] 8.1 Create or update `99_Metadata/status.json` for each managed run
  - [x] 8.2 Populate supersession fields where replacements exist

- [x] 9. Build experiment indexes
  - [x] 9.1 Generate `experiments/<slug>/index.json`
  - [x] 9.2 Include parity summary fields when available

- [x] 10. Update pointer files
  - [x] 10.1 Set `latest_completed.txt`
  - [x] 10.2 Set `canonical_acceptance.txt`
  - [x] 10.3 Set `best_saved_batch.txt`

- [x] 11. Cleanup
  - [x] 11.1 Remove empty directories created by moves
  - [x] 11.2 Leave non-empty legacy directories unless explicitly superseded
  - [x] 11.3 Require explicit allow-list for non-empty deletion

## Phase 5: Documentation and Validation

- [x] 12. Update documentation
  - [x] 12.1 Update `docs/reference/core/COMPARISON_LAYOUT.md` with experiment/pointer layer
  - [x] 12.2 Update chapter docs to prefer pointer references where appropriate

- [x] 13. Validate references and integrity
  - [x] 13.1 Verify all pointer targets exist
  - [x] 13.2 Verify chapter/reports links resolve
  - [x] 13.3 Verify no run lost required staged folders
  - [x] 13.4 Run stale-reference grep gate after rename/move operations

- [x] 14. Operational check
  - [x] 14.1 Run a one-off normalization dry run report
  - [x] 14.2 Confirm cleanup candidates are only failed/superseded runs
  - [x] 14.3 Confirm migration report captures applied changes and conflicts

## Phase 6: Weakness-Driven Hardening

- [x] 15. Address known command execution fragility
  - [x] 15.1 Implement short, script-driven commands instead of long one-liners
  - [x] 15.2 Add deterministic logging for each migration phase

- [x] 16. Address known stale-reference risk
  - [x] 16.1 Add doc-link verification helper for `docs/` and chapter references
  - [x] 16.2 Block finalization when unresolved moved-path references remain

- [x] 17. Address known cleanup overreach risk
  - [x] 17.1 Require explicit approval artifact for destructive cleanup
  - [x] 17.2 Record deletion justifications in migration report

## Deliverables

- [x] `requirements.md` finalized
- [x] `design.md` finalized
- [x] `tasks.md` finalized
- [x] experiment and pointer scaffolding created
- [x] inventory report for current runs
- [x] applied consolidation with updated docs and verified links
- [x] migration report with conflict and deletion audit trail
