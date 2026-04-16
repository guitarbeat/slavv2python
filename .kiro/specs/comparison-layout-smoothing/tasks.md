# Implementation Plan: Comparison Layout Smoothing

## Overview

This plan implements the new comparison layout organization model with minimal
risk and full backward compatibility for existing staged runs.

Repository scan grounding for this plan:

- direct staged roots: `20260327_150656_clean_parity`,
  `20260327_161610_clean_python_full`, `20260330_parity_full_postfix`,
  `20260330_cross_compare_postfix`, `20260401_live_parity_retry`,
  `20260413_release_verify`
- aggregate roots with `run_*` children: `20260328_023500_matlab_consistency`,
  `20260328_142659_python_consistency`, `20260330_python_consistency_postfix`

## Phase 1: Spec and Schema Finalization

- [ ] 1. Confirm run lifecycle schema
  - [ ] 1.1 Finalize `99_Metadata/status.json` required fields and enums
  - [ ] 1.2 Document state transition rules
  - [ ] 1.3 Finalize aggregate-root rollup policy for `run_*` containers

- [ ] 2. Confirm experiment slug taxonomy
  - [ ] 2.1 Define initial slug set (`release-verify`, `saved-batch`, `python-full`, `postfix-parity`, `postfix-cross-compare`, `python-consistency`, `matlab-consistency`, `live-parity`)
  - [ ] 2.2 Document slug assignment rules for new runs
  - [ ] 2.3 Freeze initial mapping for currently retained run roots

## Phase 2: Filesystem Scaffolding

- [ ] 3. Create root scaffolding
  - [ ] 3.1 Create `slavv_comparisons/experiments/`
  - [ ] 3.2 Create `slavv_comparisons/pointers/`

- [ ] 4. Seed pointer files
  - [ ] 4.1 Create `latest_completed.txt`
  - [ ] 4.2 Create `canonical_acceptance.txt`
  - [ ] 4.3 Create `best_saved_batch.txt`

## Phase 3: Non-Destructive Inventory

- [ ] 5. Implement inventory script/report in `dev/scripts/maintenance/`
  - [ ] 5.1 Detect current run roots and staged completeness
  - [ ] 5.2 Extract run status from `run_snapshot.json` and analysis artifacts
  - [ ] 5.3 Detect root shape (`direct` vs `aggregate`)
  - [ ] 5.4 Propose date-first renames where needed
  - [ ] 5.5 Propose experiment slug assignment per run
  - [ ] 5.6 Emit machine-readable dry-run report (moves, skips, conflicts, delete candidates)

- [ ] 6. Validate inventory output manually
  - [ ] 6.1 Confirm canonical runs are marked `retention=keep`
  - [ ] 6.2 Confirm failed runs have replacement linkage before cleanup eligibility
  - [ ] 6.3 Confirm aggregate roots have consistent rollup status behavior

## Phase 4: Apply Consolidation

- [ ] 7. Apply grouping and naming updates
  - [ ] 7.1 Move runs into `experiments/<slug>/runs/`
  - [ ] 7.2 Apply date-first rename normalization
  - [ ] 7.3 Preserve run-internal staged folder structure
  - [ ] 7.4 Support idempotent re-run after partial migration

- [ ] 8. Write status metadata
  - [ ] 8.1 Create or update `99_Metadata/status.json` for each managed run
  - [ ] 8.2 Populate supersession fields where replacements exist

- [ ] 9. Build experiment indexes
  - [ ] 9.1 Generate `experiments/<slug>/index.json`
  - [ ] 9.2 Include parity summary fields when available

- [ ] 10. Update pointer files
  - [ ] 10.1 Set `latest_completed.txt`
  - [ ] 10.2 Set `canonical_acceptance.txt`
  - [ ] 10.3 Set `best_saved_batch.txt`

- [ ] 11. Cleanup
  - [ ] 11.1 Remove empty directories created by moves
  - [ ] 11.2 Leave non-empty legacy directories unless explicitly superseded
  - [ ] 11.3 Require explicit allow-list for non-empty deletion

## Phase 5: Documentation and Validation

- [ ] 12. Update documentation
  - [ ] 12.1 Update `docs/reference/COMPARISON_LAYOUT.md` with experiment/pointer layer
  - [ ] 12.2 Update chapter docs to prefer pointer references where appropriate

- [ ] 13. Validate references and integrity
  - [ ] 13.1 Verify all pointer targets exist
  - [ ] 13.2 Verify chapter/reports links resolve
  - [ ] 13.3 Verify no run lost required staged folders
  - [ ] 13.4 Run stale-reference grep gate after rename/move operations

- [ ] 14. Operational check
  - [ ] 14.1 Run a one-off normalization dry run report
  - [ ] 14.2 Confirm cleanup candidates are only failed/superseded runs
  - [ ] 14.3 Confirm migration report captures applied changes and conflicts

## Phase 6: Weakness-Driven Hardening

- [ ] 15. Address known command execution fragility
  - [ ] 15.1 Implement short, script-driven commands instead of long one-liners
  - [ ] 15.2 Add deterministic logging for each migration phase

- [ ] 16. Address known stale-reference risk
  - [ ] 16.1 Add doc-link verification helper for `docs/` and chapter references
  - [ ] 16.2 Block finalization when unresolved moved-path references remain

- [ ] 17. Address known cleanup overreach risk
  - [ ] 17.1 Require explicit approval artifact for destructive cleanup
  - [ ] 17.2 Record deletion justifications in migration report

## Deliverables

- [ ] `requirements.md` finalized
- [ ] `design.md` finalized
- [ ] `tasks.md` finalized
- [ ] experiment and pointer scaffolding created
- [ ] inventory report for current runs
- [ ] applied consolidation with updated docs and verified links
- [ ] migration report with conflict and deletion audit trail
