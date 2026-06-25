---
title: Edge watershed parity — seeds=2 and no conflict painting on the exact route
module: pipeline/edges
tags: [edges, watershed, get_edges_V300, choose_edges, parity]
problem_type: parity
resolution_type: code_fix
---

# Edge watershed parity — seeds=2 and no conflict painting

## Problem
First `prove-exact --stage edges` vs `180709_E_crop_M_v2` showed a large
divergence: MATLAB 15,511 edges vs Python 9,429 (only 5,109 shared pairs).

## Evidence
- Pair-set diff (normalized connections): 5,109 shared, 10,402 MATLAB-only,
  4,320 Python-only.
- MATLAB `get_edges_V300.m:100` hard-codes `edge_number_tolerance = 2`.
- Run `validated_params` had `edge_number_tolerance: 4` and
  `comparison_exact_network_use_conflict_painting: True` (paper profile).
- MATLAB `vectorize_V200.m:3633-3643` has `choose_edges_V200` fully commented out.

## Root Cause
Two MATLAB-faithfulness bugs in the exact route:
1. The watershed seed count honored the param (4) instead of MATLAB's hard-coded
   `edge_number_tolerance = 2` (`number_of_edges_per_vertex=4` is only for the
   later degree-excess cleanup). Double seeds perturbed the whole watershed.
2. The exact route ran conflict painting (`choose_edges_V200`-style), which MATLAB
   does not run at all on this route.

## Solution
- `matlab_get_edges_by_watershed.py`: hard-code `edge_number_tolerance = 2`
  (the `get_edges_V300` contract), ignoring the param like MATLAB does.
- `selection.py`: force `use_conflict_painting = False` when
  `comparison_exact_network` is set; honor the param only for non-exact runs.

## Verification
`resume-exact-run --force-rerun-from edges --stop-after edges` then
`prove-exact --stage edges` vs `180709_E_crop_M_v2`: Python **9,429 → 13,775**
edges, shared **5,109 → 8,135**. 23 edge unit tests pass. Commit `3bc4a5e8`.

## Follow-Up
Edges are NOT fully closed (~52% shared). The residual is **early trace
termination on long paths** (5,984/7,376 missing edges are never traced, mean
length 9.1 vs 5.3 shared). Frontier ordering, the size-penalty reference scale,
and the `microns_per_voxel` ordering were each tested and ruled out (the last
regressed). Next: single-edge trace instrumentation, not parameter guesses.
