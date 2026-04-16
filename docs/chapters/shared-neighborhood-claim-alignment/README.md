# Shared Neighborhood Claim Alignment

Active chapter opened on April 10, 2026.

This chapter starts from a narrower parity framing than the previous one.
The recurring artifact-level clue is no longer just "Python is missing some
candidate pairs." The stronger pattern is wrong local partner choice, claim
ordering, and branch invalidation around shared active neighborhoods.

## Why This Chapter Exists

- Chapter 1 proved exact vertex parity on the imported-MATLAB surface and exact
  downstream `network` parity when exact MATLAB `edges` are imported.
- Chapter 2 narrowed the saved-batch edge gap to `94/93` and established that
  the active MATLAB target is `get_edges_V300 -> get_edges_by_watershed`.
- The next missed clue is neighborhood-scoped: at vertices like `359`, `866`,
  and `1283`, missing MATLAB pairs never reach the Python candidate manifest
  even though alternate nearby partners remain active.
- The repeated `terminal_frontier_hit > valid frontier connection count`
  pattern suggests local invalidation or reassignment before chooser cleanup,
  not just missing global candidate discovery.

## Starting Facts

- Current best retained saved-batch result is `vertices 110/110`,
  `edges 94/93`, `strands 49/54`.
- The live rerun is still farther off: `1379` MATLAB edges vs `1425` Python
  edges, and `682` MATLAB strands vs `681` Python strands.
- The imported-MATLAB workflow is the active finish line for exact parity in
  this phase; native Python-from-energy parity is still out of scope.
- Origin `64` remains useful, but it is now a neighborhood probe rather than
  the whole story.
- Shared neighborhoods around `359`, `866`, and `1283` show stronger evidence
  of partner substitution and claim-order drift than a simple nearest-neighbor
  explanation.
- Candidate coverage is still the first triage metric, but candidate counts
  alone are not enough to explain final parity.

## Main Goal

Make Python admit, retain, and resolve candidate relationships across shared
neighborhoods the same way MATLAB `get_edges_by_watershed` does before
the parity cleanup chain runs.

## Working Questions

1. At the worst shared neighborhoods, which branches are explored, invalidated,
   reassigned, or dropped before they ever reach the candidate manifest?
2. Does MATLAB temporarily allow over-budget local candidate states that Python
   suppresses too early?
3. Are bifurcation-half choice and parent/child invalidation changing which
   branch survives at a shared vertex?
4. Which shared-map surfaces such as `vertex_index_map`, `pointer_map`, branch
   bookkeeping, and claim ordering are still not represented in Python?
5. What is the smallest isolated regression test that captures one real
   neighborhood-level mismatch?

## Scope

In scope:

- neighborhood-level candidate lifecycle diagnostics
- claim ownership and claim ordering at shared vertices
- local invalidation before candidate manifest emission
- bifurcation-half choice and parent/child cleanup when they change incident
  pairs
- saved-batch experiments that preserve imported-MATLAB vertex parity

Out of scope:

- re-solving vertex parity
- broad global threshold sweeps without local evidence
- downstream network assembly changes that already pass the stage-isolated
  exact-MATLAB-edges gate
- blanket ownership rules like the rejected full-path-claim experiment

## First Loop

Use the saved-batch surface as the lab, but inspect neighborhoods instead of
only aggregate counts.

1. Reuse a staged saved-batch run root.
2. Record per-neighborhood candidate lifecycle artifacts for `64`, `359`,
   `866`, and `1283`.
3. Compare those artifacts against MATLAB's shared-map semantics, not just
   final endpoint pairs.
4. Promote a change only if the saved-batch surface improves without regressing
   the stage-isolated `network` gate.
5. Treat the live MATLAB-backed rerun as the release-grade acceptance surface
   before claiming imported-MATLAB parity is complete.

```powershell
python dev/scripts/cli/compare_matlab_python.py `
  --input C:\slavv_comparisons\saved_batch_run\01_Input\synthetic_branch_volume.tif `
  --skip-matlab `
  --output-dir C:\slavv_comparisons\shared_neighborhood_claim_trial `
  --params C:\slavv_comparisons\saved_batch_run\99_Metadata\comparison_params.normalized.json
```

## Deliverables For This Chapter

- A diagnostic artifact that shows admission, invalidation, reassignment, and
  final survival per shared neighborhood.
- At least one isolated regression test for a real neighborhood-level mismatch.
- One targeted Python change that improves the imported-MATLAB saved-batch loop
  without causing a larger regression on the live confirmation surface.
- Updated run docs that cite the canonical saved-batch lab, the canonical live
  acceptance root, and the exact commands for rerun, diagnostics, and proof
  inspection.

## Working Docs

- [Shared Neighborhood Claim Investigation Plan](INVESTIGATION_PLAN.md)
- [Shared Neighborhood Audit Checklist](NEIGHBORHOOD_AUDIT_CHECKLIST.md)
- [Parity Workflow Completion Spec Archive](parity-workflow-completion-spec/tasks.md)

## Core References

- [MATLAB Translation Guide](../../reference/MATLAB_TRANSLATION_GUIDE.md)
- [Comparison Run Layout](../../reference/COMPARISON_LAYOUT.md)
- [Shared Candidate Generation Alignment](../shared-candidate-generation/README.md)
- [Imported-MATLAB Parity Report](../imported-matlab-parity/PARITY_REPORT_2026-04-09.md)
- [Parity Findings 2026-03-27](../imported-matlab-parity/PARITY_FINDINGS_2026-03-27.md)
- [MATLAB Parity Audit Checklist](../imported-matlab-parity/MATLAB_PARITY_AUDIT_CHECKLIST.md)

