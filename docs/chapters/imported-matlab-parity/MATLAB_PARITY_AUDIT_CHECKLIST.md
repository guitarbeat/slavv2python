# MATLAB Parity Audit Checklist

This checklist turns the current MATLAB-vs-Python parity discussion into a
concrete audit plan we can work through and check off.

This file is intentionally a working checklist, not the general parity status
or workflow plan.

## Rapid Recall

- Use this file after the stage-isolated `network` gate passes and you are
  back on the remaining `edges` problem.
- The current leading diagnosis is not "rewrite the algorithm again."
- The current leading diagnosis is "audit the small local MATLAB-vs-Python
  frontier decisions that decide which candidate pairs ever enter the pool."
- The highest-value sections remain:
  - frontier voxel admission and overwrite
  - terminal-hit ownership semantics
  - parent/child and bifurcation claim ordering
  - post-hit pruning and local partner substitution

## Read This File When

- you are comparing `get_edges_for_vertex.m` to `tracing.py`
- you want a concrete check-off audit rather than a narrative findings doc
- you need to know which local semantic differences are still worth auditing

## Goal

- [ ] Confirm that Python makes the same local frontier decisions as MATLAB in
      the edge-building path.
- [ ] Narrow the remaining parity gap by auditing implementation semantics, not
      by rewriting the high-level algorithm again.

## Why Exact Parity Is Hard

- [ ] Remember that the MATLAB code is a control implementation, not a formal
      spec. It tells us what happens, but not always which behaviors are
      essential versus incidental.
- [ ] Treat parity as an implementation-emulation problem, not just an
      algorithm-translation problem.
- [ ] Keep in mind that small early differences in frontier expansion can
      change which candidate edges are ever generated, which then changes every
      later stage.
- [ ] Assume library-level equivalence is not enough. MATLAB and Python can use
      equivalent operations that still differ in exact voxel-level behavior.

## Highest-Priority Audit Categories

### 1. Frontier Voxel Admission And Pointer Overwrite Rules

- [x] Verify that Python matches MATLAB's strict "new index" test exactly.
- [x] Verify that Python writes pointer and distance state at the same point in
      the loop that MATLAB does.
- [x] Verify default-map behavior for unseen voxels matches MATLAB sparse-array
      semantics closely enough for parity.
- [ ] Compare:
      [get_edges_for_vertex.m](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\external\Vectorization-Public\source\get_edges_for_vertex.m)
      around line 202 and
      [tracing.py](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\source\slavv\core\tracing.py)
      around line 1523.
- [ ] Add a focused parity note here if any admission or overwrite mismatch is
      found.

Current findings:

- MATLAB sets `pointer_energy_map(current_index) = -Inf` before testing
  neighbors, and Python does the same with
  `pointer_energy_map[current_linear] = float("-inf")`.
- MATLAB admits a neighbor only when
  `pointer_energy_map(current_linear_strel) > energy_map(current_index)`.
  Python mirrors that strict comparison with
  `pointer_energy_map.get(linear_index, 0.0) > current_energy`.
- MATLAB sparse maps return zero for unseen entries; Python mirrors that with
  `dict.get(..., 0.0)`, which is the relevant default under the imported-MATLAB
  negative-energy parity path.
- Both implementations write pointer, pointer-energy, and distance state before
  applying the length cutoff. That means a voxel can be recorded in pointer
  state even if it is later excluded from the available frontier.
- Static code reading did not show an obvious admission-rule mismatch. This
  section stays open only for artifact-level tracing on the worst shared
  vertices.

### 2. Next-Voxel Selection And Equal-Energy Tie Breaks

- [x] Prove that Python's heap ordering matches MATLAB's `min` behavior on the
      sparse available-energy surface.
- [x] Prove that equal-energy voxels break ties in the same MATLAB linear-index
      order.
- [x] Check whether stale entries in the Python heap can change traversal order
      compared with MATLAB's repeated `min` over the current sparse image.
- [ ] Compare:
      [get_edges_for_vertex.m](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\external\Vectorization-Public\source\get_edges_for_vertex.m)
      around line 425 and
      [tracing.py](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\source\slavv\core\tracing.py)
      around line 1621.
- [x] Add a targeted regression case if we find a tie-breaking drift.

Current findings:

- MATLAB chooses the next voxel with
  `[ min_available_energy, current_index ] = min( available_energy_map )`,
  which means "lowest energy, then lowest linear index" on the current sparse
  surface.
- Python mirrors that by pushing `(available_energy, linear_index)` into the
  heap, so equal-energy ties fall back to MATLAB linear index order.
- Python also discards stale heap entries with
  `if available_map.get(candidate_linear) != candidate_energy: continue`,
  which keeps the heap aligned with MATLAB's repeated `min` over the current
  frontier state rather than over historical entries.
- Unit coverage already exists in
  [test_frontier_tracing.py](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\tests\unit\core\test_frontier_tracing.py)
  for deterministic tie-breaking and frontier-budget behavior, and those tests
  passed on April 7, 2026.
- No tie-breaking drift has been identified yet, so this item is currently
  marked as aligned by static inspection plus regression coverage.

### 3. Terminal-Hit Ownership Semantics

- [x] Verify that Python backtracks and stamps ownership on the recovered path
      in the same way MATLAB does.
- [ ] Verify the exact role of the root or bifurcation voxel after a terminal
      hit.
- [ ] Confirm which voxels should remain claimable by later paths and which
      should become owned immediately.
- [x] Keep the failed blanket full-path-claim experiment in mind and avoid
      repeating it as-is.
- [ ] Compare:
      [get_edges_for_vertex.m](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\external\Vectorization-Public\source\get_edges_for_vertex.m)
      around line 271 and
      [tracing.py](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\source\slavv\core\tracing.py)
      around line 1566.
- [ ] Record whether the next fix needs selective ownership or selective claim
      ordering rather than broader ownership.

Current findings:

- Python backtracks from the terminal hit through `pointer_index_map` and
  records the recovered path before storing the edge, which is the same overall
  control shape MATLAB uses.
- Unit coverage already exists for path backtracking and "terminal hit clears
  new frontier voxels" behavior in
  [test_frontier_tracing.py](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\tests\unit\core\test_frontier_tracing.py),
  and those tests passed on April 7, 2026.
- The unresolved question is not whether Python backtracks at all, but exactly
  how ownership should treat the shared root or bifurcation voxel after a
  terminal hit.
- The full-path-claim experiment already showed that blanket ownership is too
  aggressive: it improved one local shared-vertex signal but made overall edge
  and strand parity materially worse.
- The next fix should therefore be selective. The open choice is whether that
  selectivity belongs in ownership itself or in subsequent claim ordering.
- Artifact-level clue from
  [comparison_report.json](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\comparisons\20260406_conflict_provenance_trial\comparison_report.json)
  and the staged `units/` payloads:
  - origin `359` recorded `terminal_frontier_hit = 2` but only one valid
    frontier connection in its unit payload
  - origin `866` recorded `terminal_frontier_hit = 3` but only one valid
    frontier connection in its unit payload
  - origin `1283` recorded `terminal_frontier_hit = 3` but only two valid
    frontier connections in its unit payload
- That means some shared-vertex branches are being invalidated or reassigned
  before they ever reach the candidate manifest, which keeps this section high
  priority.

### 4. Parent/Child Invalidation And Bifurcation-Half Choice

- [x] Verify that Python invalidates children under the same conditions as
      MATLAB.
- [x] Verify that Python computes parent and child energy over the same voxel
      sets as MATLAB.
- [ ] Verify that the bifurcation point is identified at the same path index.
- [ ] Verify that the origin-half selection rule matches MATLAB when one half
      is empty or when half energies tie.
- [ ] Compare:
      [get_edges_for_vertex.m](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\external\Vectorization-Public\source\get_edges_for_vertex.m)
      around line 308 and
      [tracing.py](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\source\slavv\core\tracing.py)
      around line 1363.
- [ ] Add a parity note if the drift is really in half selection rather than in
      path generation.

Current findings:

- Python mirrors MATLAB's "child better than parent means invalidate the child"
  rule with a strict energy comparison, and unit coverage exists for both the
  rejected-child and accepted-child cases.
- Static code reading shows both implementations compute parent and child
  energy as the maximum sampled energy along each path segment.
- The remaining open part of this section is narrower: we still need artifact-
  level proof that Python identifies the same bifurcation index and chooses the
  same parent half around the worst shared vertices.
- If a mismatch appears here, it is more likely to be a path-membership or
  bifurcation-location issue than a high-level invalidation-rule issue.
- The `terminal_frontier_hit > valid frontier connection count` pattern at
  origins `359`, `866`, and `1283` is consistent with this section still being
  a live suspect. The missing branches may be getting invalidated locally
  rather than never being explored at all.

### 5. Post-Hit Pruning Beyond Found Vertices

- [x] Verify that Python prunes newly exposed frontier voxels with the same
      displacement-vector math MATLAB uses.
- [x] Verify that the prune happens at the same point in the control flow.
- [x] Verify that only the intended frontier voxels are cleared after a
      terminal hit.
- [ ] Compare:
      [get_edges_for_vertex.m](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\external\Vectorization-Public\source\get_edges_for_vertex.m)
      around line 221 and
      [tracing.py](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\source\slavv\core\tracing.py)
      around line 1337.
- [ ] Note whether the missing MATLAB pairs are being suppressed here before
      cleanup ever sees them.

Current findings:

- Python uses the same normalized displacement-vector projection test that
  MATLAB uses to prune frontier voxels lying beyond an already-found terminal
  direction.
- Unit coverage exists for both the geometric prune helper and the
  post-terminal frontier-clearing behavior, and those tests passed on April 7,
  2026.
- Static code reading suggests the prune is in the right part of the control
  flow: new frontier voxels are admitted, length-filtered, optionally pruned,
  and then cleared after a terminal hit instead of being left queued.
- This section remains open only because the live parity artifacts still need
  vertex-by-vertex proof of whether the missing MATLAB candidate pairs are
  being suppressed here or even earlier.
- Current artifact read on the top shared vertices points slightly upstream of
  pure pruning:
  - the missing MATLAB pairs are absent from the candidate manifest
  - but their partner vertices are still active elsewhere in the graph, often
    with chosen alternate neighbors
  - that makes "local partner substitution around a shared neighborhood" a
    better immediate hypothesis than "the partner region was never reachable"

### 6. Border And Length Cutoffs

- [x] Verify that Python matches MATLAB's border-entry rule for whether the
      frontier loop begins at all.
- [x] Verify the off-by-one seed on `distance_map`.
- [x] Verify the strict `< max_edge_length_in_microns` cutoff.
- [ ] Verify that scale-derived edge-length budgets are computed from the same
      scale source and the same rounding rules.
- [ ] Compare:
      [get_edges_for_vertex.m](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\external\Vectorization-Public\source\get_edges_for_vertex.m)
      around lines 35, 91, and 104 and
      [tracing.py](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\source\slavv\core\tracing.py)
      around lines 1443 and 1490.
- [ ] Log any vertex-specific cases where Python stops early or explores too
      long.

Current findings:

- MATLAB enters the edge-search loop only when the origin is far enough from
  the image boundary for the current structuring element. Python mirrors that
  by returning early when any origin coordinate lies within `strel_apothem` of
  a border.
- MATLAB seeds `distance_map(current_index) = 1` before expansion. Python
  mirrors that with `distance_map = {origin_linear: 1.0}` and carries that
  forward into new-neighbor distances.
- MATLAB keeps only candidates with
  `distance_map(new_indices_considered) < max_edge_length_in_microns`. Python
  uses the same strict `<` cutoff after counting rejected candidates.
- The remaining open part is scale provenance rather than the cutoff math
  itself: MATLAB reads the scale from the HDF5 energy volume at the current
  vertex, while Python currently consumes imported `vertex_scales`.
- Spot checks on the worst shared vertices matched between Python
  `vertex_scales` and the raw MATLAB HDF5 scale channel after normalization,
  but this is not yet a global proof.

## Lower-Priority Audit Category

### 7. Scale Sourcing And Rounding

- [ ] Confirm globally that imported Python `vertex_scales` match the raw
      MATLAB HDF5 scale channel after the expected normalization.
- [ ] Confirm that any rounding or clipping of vertex positions happens the
      same way before scale lookup.
- [ ] Keep this lower priority unless new evidence says otherwise, because the
      worst shared vertices already matched in spot checks.
- [ ] Compare:
      [get_edges_for_vertex.m](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\external\Vectorization-Public\source\get_edges_for_vertex.m)
      around line 24 and
      [tracing.py](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\source\slavv\core\tracing.py)
      around line 1463.

Current findings:

- The current evidence does not point to a scale mismatch as the main blocker.
- Spot checks on shared vertices `359`, `866`, and `1283` matched between the
  staged Python checkpoint scales and the raw MATLAB HDF5 scale channel after
  the expected minus-one normalization.
- This section stays open because "spot checked" is not the same as "proved
  globally," but it is not the best next place to spend debugging time.

## Deprioritized For Now

- [ ] Do not start with neighborhood offset order or MATLAB linear-index
      conversion unless new evidence appears. Those already look aligned in
      [tracing.py](C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\source\slavv\core\tracing.py)
      around line 944.

## Working Notes

- [ ] Top shared vertices from the current parity investigations:
      `359`, `866`, `1283`.
- [ ] Current working hypothesis: the remaining mismatch is upstream in frontier
      candidate generation, not only in downstream cleanup.
- [ ] Current practical goal: prove that Python makes the same local frontier
      decisions in the same order as MATLAB.

Artifact-level notes from the April 6, 2026 conflict-provenance trial:

- Vertex `359`
  - origin `359` contributed only `[359, 181]` from the frontier path
  - the chosen extra frontier edges touching `359` came from neighboring
    origins `1180` and `1568`
  - missing MATLAB incident pairs from the report included `[359, 1046]`,
    `[359, 1284]`, and `[359, 1300]`
  - partner vertices `1284` and `1300` were still active elsewhere and had
    chosen alternate partners, which points to local partner substitution
- Vertex `866`
  - origin `866` recorded `terminal_frontier_hit = 3` but only yielded the
    frontier candidate `[866, 885]`
  - missing MATLAB incident pairs from the report included `[866, 1023]`,
    `[866, 1203]`, and `[866, 1348]`
  - partner vertices `1023`, `1203`, and `1348` were active elsewhere, with
    chosen or candidate edges to other neighbors
- Vertex `1283`
  - origin `1283` yielded `[1283, 1134]`, `[1283, 768]`, and watershed
    candidate `[1283, 1659]`
  - missing MATLAB incident pairs from the report included `[95, 1283]`,
    `[542, 1283]`, and `[1283, 1319]`
  - partner vertices `95` and `542` had chosen alternate partners, and `1319`
    was active in nearby candidate structure as well
- Shared conclusion:
  - the first artifact-level divergence looks like wrong local partner choice
    around active neighborhoods, not dead regions or global frontier failure
  - geometry alone does not fully explain it:
    - at `866`, the chosen alternatives (`885`, `810`) are clearly closer than
      the missing MATLAB partners, which is consistent with local competition
    - at `1283`, missing partner `1319` is roughly as close as chosen partners
      `1659` and `1134`, so the remaining gap is not just "Python picked the
      nearest neighbor"

## Completion Criteria

- [ ] We can explain the missing MATLAB candidate pairs at the worst shared
      vertices in terms of a specific local semantic mismatch.
- [ ] We have at least one regression test that reproduces the discovered
      mismatch in isolation.
- [ ] A targeted Python change improves parity on the imported-MATLAB loop
      without causing a larger regression elsewhere.
