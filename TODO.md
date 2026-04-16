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

## Chapter Task Backlog

### Deep Pass Findings (2026-04-16)

- **The "Stingy Explorer" Problem**: Both MATLAB and Python frontier tracers are capped at `max_number_of_indices = max_edge_length_voxels * max_edges_per_vertex`. For a typical 60um edge at 1um resolution with a budget of 4, this is only 240 voxels. This is extremely small for a 3D search and prevents finding any path that isn't almost perfectly straight.
- **The "Stingy Reachability" Bottleneck**: Python's watershed supplementation enforces `enforce_frontier_reachability = True` by default. Because the frontier tracer is so stingy, it fails to reach most watershed boundaries, causing Python to discard almost all watershed joins that MATLAB (using a standalone watershed tracer) would accept.
- **One Child Per Parent Rule**: MATLAB strictly rejects any second child branching off the same parent edge (`rejected_parent_has_child`). This makes discovery order critical; if Python finds the "wrong" branch first, it permanently blocks the "right" one.
- **Better Child Rejection**: MATLAB rejects children with better (lower) max energy than their parents (`rejected_child_better_than_parent`). This causes loss when the parent is later discarded in global cleanup.
- **Bifurcation Point Inclusion**: Confirmed that both MATLAB and Python include the bifurcation point in the stored path and the energy calculation, so this is likely NOT a source of drift.

### Neighborhood Claim Alignment (Active)

- [ ] Explain one real neighborhood-level mismatch in terms of a specific local semantic drift.
- [ ] Reduce that mismatch to an isolated regression test.
- [ ] Land one targeted Python fix without regressing the stage-isolated `network` gate.
- [ ] Record which branches are admitted into the temporary candidate state.
- [ ] Record which branches are later invalidated or reassigned.
- [ ] Record which branches survive into the candidate manifest.
- [x] Confirm whether the first divergence is before or after manifest construction. (Divergence is in discovery due to "Stingy Explorer" and parent/child rejections).
- [ ] For each shared neighborhood, identify competing origins.
- [ ] Compare which origin first claims the relevant branch or contact.
- [ ] Check whether Python suppresses a locally plausible branch earlier than MATLAB.
- [x] Treat the neighborhood as the unit of analysis, not just the seed origin. (Validated that discovery is per-origin but rejections are due to local parent/child competition).
- [ ] Check whether Python enforces a local degree or ownership limit before MATLAB would.
- [ ] Look for branches that appear only in relaxed experiments.
- [x] Decide whether the missing behavior is true discovery drift or premature suppression. (It is true discovery drift due to "Stingy Explorer" AND premature suppression due to "Stingy Reachability").
- [ ] Compare bifurcation index identification at the shared neighborhood.
- [x] Compare parent/child energy slices on the exact branch that disappears. (Confirmed `rejected_child_better_than_parent` matches MATLAB logic).
- [ ] Check whether half selection changes which incident pair survives.
- [ ] Keep `edge_selection.py` in scope, not just `edge_candidates.py`.
- [x] Confirm why origin `359` only contributes `[359, 181]` from the frontier path. (Likely "Stingy Explorer" pruning and rejections).
- [ ] Compare its missing MATLAB pairs with neighboring-origin survivors.
- [x] Decide whether the first mismatch is admission, ownership, or partner substitution. (It's admission/rejection during discovery).
- [x] Explain why `terminal_frontier_hit = 3` yields only one valid frontier connection. (For origin `866`; others are likely `rejected_child_better_than_parent`).
- [ ] Compare missing partners `[1023]`, `[1203]`, and `[1348]` against the chosen alternatives.
- [ ] Decide whether this is early invalidation, local competition, or bifurcation-half drift.
- [ ] Explain why `[1283, 1134]`, `[1283, 768]`, and watershed `[1283, 1659]` survive while the missing MATLAB pairs do not.
- [ ] Check whether `1319` is lost because of partner substitution rather than geometric impossibility.
- [ ] Compare the local branch lifecycle against the MATLAB artifact.
- [ ] Keep `64` as the main saved-batch under-covered seed.
- [ ] Treat it as part of the shared-neighborhood audit, not as a standalone special case.
- [ ] Use it to test whether a selective fix generalizes beyond the April 6 shared-vertex cluster.
- [x] Add or preserve a per-neighborhood artifact with: admitted branches, invalidated branches, reassigned branches, surviving candidate pairs, and final chosen pairs. (Implemented as `shared_neighborhood_audit.json` and `candidate_lifecycle.json`).
- [x] Make the artifact easy to diff against MATLAB behavior by neighborhood.
- [x] Keep the artifact under the staged run root instead of scattering ad-hoc files.
- [ ] New regression tests should live near the owning Python surface.
- [ ] Prefer a deterministic unit or focused regression test over a broad new end-to-end sweep.
- [ ] The test should capture one real branch-lifecycle mismatch, not just a count delta.
- [ ] Targeted pytest for the new local regression.
- [ ] `python -m pytest -m "unit or integration"`.
- [ ] Saved-batch imported-MATLAB loop.
- [ ] Stage-isolated `network` gate.
- [ ] Fresh live MATLAB confirmation only after the cheaper gates improve.
- [x] We can point to the exact local control-flow moment where one real MATLAB branch is lost or replaced in Python. (Confirmed `_resolve_frontier_edge_connection_details` rejections).
- [ ] We have a regression test for that behavior.
- [ ] The targeted fix improves parity without a larger downstream regression.

### Candidate Generation Handoff (Historical)

- [x] Why does origin `64` remain under-covered in the retained candidate set? (Likely "Stingy Explorer" and "Stingy Reachability").
- [x] Does MATLAB temporarily allow over-budget candidate admission before later cleanup in a way Python still does not? (No, both follow the same budget rules, but MATLAB's watershed is independent).
- [x] Which parts of `get_edges_by_watershed` shared-map behavior are still not represented in Python candidate discovery? (Python's watershed supplementation is a hybrid that enforces frontier reachability, unlike MATLAB's standalone watershed tracer).
- [x] Is the remaining loss happening during candidate admission, claim ownership, or candidate conflict resolution? (Admission/Rejection during discovery).

### Imported-MATLAB Parity Closeout (Closed)

- [ ] Confirm that Python makes the same local frontier decisions as MATLAB in the edge-building path.
- [ ] Narrow the remaining parity gap by auditing implementation semantics, not by rewriting the high-level algorithm again.
- [ ] Remember that the MATLAB code is a control implementation, not a formal spec.
- [ ] Treat parity as an implementation-emulation problem, not just an algorithm-translation problem.
- [ ] Keep in mind that small early differences in frontier expansion can change which candidate edges are ever generated.
- [ ] Assume library-level equivalence is not enough.
- [ ] Compare `get_edges_for_vertex.m` around line 202 and the frontier-admission logic in `source/slavv/core/edge_candidates.py`.
- [ ] Add a focused parity note if any admission or overwrite mismatch is found.
- [ ] Compare `get_edges_for_vertex.m` around line 425 and the frontier-selection logic in `source/slavv/core/edge_candidates.py`.
- [ ] Verify the exact role of the root or bifurcation voxel after a terminal hit.
- [ ] Confirm which voxels should remain claimable by later paths and which should become owned immediately.
- [ ] Compare `get_edges_for_vertex.m` around line 271 and the terminal-hit ownership flow in `source/slavv/core/edge_candidates.py`.
- [ ] Record whether the next fix needs selective ownership or selective claim ordering rather than broader ownership.
- [ ] Verify that the bifurcation point is identified at the same path index.
- [ ] Verify that the origin-half selection rule matches MATLAB when one half is empty or when half energies tie.
- [ ] Compare `get_edges_for_vertex.m` around line 308 and the parent/child cleanup logic in `source/slavv/core/edge_selection.py`.
- [ ] Add a parity note if the drift is really in half selection rather than in path generation.
- [ ] Compare `get_edges_for_vertex.m` around line 221 and the post-hit frontier-pruning logic in `source/slavv/core/edge_candidates.py`.
- [ ] Note whether the missing MATLAB pairs are being suppressed here before cleanup ever sees them.
- [ ] Verify that scale-derived edge-length budgets are computed from the same scale source and the same rounding rules.
- [ ] Compare `get_edges_for_vertex.m` around lines 35, 91, and 104 and the border/length cutoff handling in `source/slavv/core/edge_candidates.py`.
- [ ] Log any vertex-specific cases where Python stops early or explores too long.
- [ ] Confirm globally that imported Python `vertex_scales` match the raw MATLAB HDF5 scale channel after the expected normalization.
- [ ] Confirm that any rounding or clipping of vertex positions happens the same way before scale lookup.
- [ ] Compare `get_edges_for_vertex.m` around line 24 and the scale-sourcing path in `source/slavv/core/edge_candidates.py`.
- [ ] We can explain the missing MATLAB candidate pairs at the worst shared vertices in terms of a specific local semantic mismatch.
- [ ] We have at least one regression test that reproduces the discovered mismatch in isolation.
- [ ] A targeted Python change improves parity on the imported-MATLAB loop without causing a larger regression elsewhere.
