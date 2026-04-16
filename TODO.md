# TODO

Active imported-MATLAB parity backlog for `slavv2python`.

This file is the short root-level answer to two questions:

- what still blocks 1:1 imported-MATLAB parity
- what previous comparison runs taught us about where to work next

## Finish Line

The finish line for this phase is exact parity on the imported-MATLAB workflow,
not native Python-from-energy parity.

Done means exact match for:

- vertices
- undirected endpoint-pair edges
- strands

Every parity claim should cite staged artifacts under:

- `01_Input/`
- `02_Output/`
- `03_Analysis/`
- `99_Metadata/`

## Canonical Run Roots

Use these as the primary evidence surfaces.

Saved-batch lab:

- `C:\Users\alw4834\Documents\slavv2python\slavv_comparisons\saved_batch_run`

Canonical live acceptance root:

- `C:\Users\alw4834\Documents\slavv2python\slavv_comparisons\release_verify_20260413\live_canonical_20260413`

## Current Baseline

As of the latest canonical live rerun:

- vertices: `2577 / 2577`
- edges: `2533 / 2463`
- strands: `1120 / 1006`

This means vertex parity is stable, but edge and strand parity are still open.

## Lessons Learned

### Run-history lessons

- March 30, 2026 was an over-emission regime, not the current failure mode.
  Python produced more edges than MATLAB.
- By April 1, 2026 the repo had already flipped into an under-emission regime.
  Python was now missing edges instead of inventing too many of them.
- April 13, 2026 is still under-emitting, but candidate coverage is much
  stronger than it was earlier in the investigation.

### What improved

- Origins `64`, `359`, and `1283` now have much richer local candidate
  coverage than they did on earlier runs.
- `1283` is no longer best understood as a pure candidate-admission problem.
  It now looks more like final cleanup loss.
- The repo is no longer blocked by missing parity infrastructure.
  The remaining work is algorithmic convergence.

### What the remaining failures actually are

The live blockers are now split into distinct classes:

- frontier pre-manifest rejection:
  `1482`, `1666`, `2129`, `2216`, `2424`, `2459`, `2486`
- manifest partner substitution:
  `1654`, `866`, `322`, `35`, `4`, `3`, `57`
- true candidate-admission gaps:
  `2305`, `2492`
- final cleanup loss:
  `1283`, `64`, `20`

### Most important current pattern

Several missing neighborhoods now show the same local shape:

1. one seed-origin frontier edge is accepted
2. one or more sibling branches are rejected as
   `rejected_child_better_than_parent`
3. the accepted parent edge is later removed during final cleanup

This pattern shows up at origins like `1482`, `2129`, `2216`, `2424`, and
`2459`.

Repeated first-divergence terminals, especially terminal `1009`, suggest that
multiple missing neighborhoods may share the same parent/child ownership or
branch invalidation rule.

## What Is Left

### 1. Fix frontier parent/child invalidation semantics

Primary target:

- `source/slavv/core/edge_candidates.py`

Why:

- the highest-leverage blockers are still `pre_manifest_rejection`
- the strongest repeated reason is `rejected_child_better_than_parent`
- multiple origins appear to share one reusable bug pattern

Focus origins:

- `1482`
- `1666`
- `2129`
- `2216`
- `2424`
- `2459`
- `2486`

Concrete goal:

- stop rejecting sibling branches too early when the accepted parent edge does
  not survive final cleanup anyway

### 2. Fix partner substitution in edge selection

Primary target:

- `source/slavv/core/edge_selection.py`

Why:

- several hotspots now have enough candidate coverage
- the first real failure is later partner choice, not missing discovery

Focus origins:

- `1654`
- `866`
- `322`
- `35`
- `4`
- `3`
- `57`

Concrete goal:

- make partner choice and claim reassignment behave like MATLAB
  `get_edges_by_watershed` at shared neighborhoods

### 3. Handle the remaining true admission gaps

Primary target:

- `source/slavv/core/edge_candidates.py`

Focus origins:

- `2305`
- `2492`

Concrete goal:

- explain why these neighborhoods have zero candidate pairs despite MATLAB
  incident pairs being present

### 4. Verify cleanup-loss cases after the earlier fixes

Check origins:

- `1283`
- `64`
- `20`

Why:

- these now look more like downstream cleanup retention issues than first-order
  candidate discovery failures
- they should be revisited after the frontier invalidation and partner-choice
  fixes

## Recommended Next Loop

1. Use the saved-batch run root for the cheapest targeted rerun.
2. Compare lifecycle artifacts for `1482` and `1666` first.
3. Patch `edge_candidates.py` only if the evidence still points to
   pre-manifest rejection.
4. Run the stage-isolated `network` gate to protect downstream exactness.
5. Promote the change only if the saved-batch surface improves.
6. Rerun the canonical live imported-MATLAB root and record counts, top
   divergence neighborhoods, and proof artifacts.
7. Move to `1654` and `866` only after the repeated frontier rejection pattern
   has been addressed.

## Useful Commands

Saved-batch rerun:

```powershell
python dev/scripts/cli/compare_matlab_python.py `
  --input C:\Users\alw4834\Documents\slavv2python\slavv_comparisons\saved_batch_run\01_Input\synthetic_branch_volume.tif `
  --skip-matlab `
  --output-dir C:\Users\alw4834\Documents\slavv2python\slavv_comparisons\shared_neighborhood_claim_trial `
  --params C:\Users\alw4834\Documents\slavv2python\slavv_comparisons\saved_batch_run\99_Metadata\comparison_params.normalized.json
```

Canonical live rerun from edges:

```powershell
python dev/scripts/cli/compare_matlab_python.py `
  --input data/slavv_test_volume.tif `
  --skip-matlab `
  --output-dir C:\Users\alw4834\Documents\slavv2python\slavv_comparisons\release_verify_20260413\live_canonical_20260413 `
  --python-parity-rerun-from edges `
  --comparison-depth deep
```

Proof artifact summary:

```powershell
slavv parity-proof --run-dir C:\Users\alw4834\Documents\slavv2python\slavv_comparisons\release_verify_20260413\live_canonical_20260413
```
