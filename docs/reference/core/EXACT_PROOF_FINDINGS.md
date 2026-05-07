# Exact Proof Findings

[Up: Reference Docs](../README.md)

**Last Updated**: 2026-05-04

This is the maintained current-status owner for the native-first exact route.
Use it for live proof status, current v22 watershed readouts, the first failing
field, and the measured effect of parity-bearing fixes.

This file is intentionally developer-facing. It does not define the acceptance
gate for the public `paper` CLI/app workflow.

Use the other core docs for different jobs:

- `MATLAB_METHOD_IMPLEMENTATION_PLAN.md` defines claim boundaries and the
  remaining roadmap.
- `MATLAB_PARITY_MAPPING.md` maps MATLAB functions to the live Python tree and
  records confirmed structural deviations.
- [v22 Pointer Corruption Archive](../../chapters/v22-pointer-corruption/README.md)
  preserves the April 2026 investigation trail and archived Kiro planning.

## Scope

- The canonical exact route is `comparison_exact_network=True` with
  `python_native_hessian` as the canonical exact-compatible energy provenance.
- Preserved MATLAB vectors remain the oracle artifacts for `prove-exact`.
- Maintained parity storage now separates preserved MATLAB truth under
  `oracles/` from disposable reruns under `runs/`.
- `100%` means artifact-level equality against preserved MATLAB vectors, not
  count-level similarity.

## Current Status

| Component | Current state | Proof state | Main blocker |
| --- | --- | --- | --- |
| Native energy | Complete | Canonical exact-compatible slavv_python | Keep MATLAB-oracle fixture coverage green |
| Vertices | Runnable on the native-first exact route | Proof pending downstream | Awaiting edge proof |
| Edges | Active parity work | Not exact | Candidate-generation and chooser control flow |
| Network | Source-aligned | Proof pending | Upstream edge parity |

## Current v22 Read

The strongest current interpretation is:

- exact-route proof runs must first pass a saved-params fairness audit before
  candidate or chooser counts are treated as trustworthy parity evidence
- the pointer-lifecycle fixes were real and should stay
- the reviewed MATLAB and Python watershed constants are already aligned

## May 2026 Trace Order Fix Validation

**Date**: 2026-05-05  
**Experiment**: `trace_order_fix`  
**Oracle**: `180709_E_batch_190910-103039`  
**Location**: `workspace\runs\trace_order_fix`
**Status**: Validation complete, leading diagnostic trail for edges

### Fix Applied
- **File**: `slavv_python/core/edges_internal/edge_selection.py`
- **Issue**: Trace order was randomized without seeded RNG, causing non-deterministic results
- **Fix**: Always use seeded RNG (`np.random.default_rng(seed)`) for trace order shuffling
- **Impact**: Ensures deterministic candidate generation for parity testing

### Results Summary (Measured on D: Drive)

| Metric | Baseline (v2.3) | trace_order_fix | Improvement |
|--------|-----------------|-----------------|-------------|
| Python candidates | 169 | 488 | 2.9x |
| Matched MATLAB pairs | 149 | 404 | 2.7x |
| Match rate | 12.4% | 33.8% | +21.4 pp |
| Missing MATLAB pairs | 1,048 | 793 | -255 pairs |
| Extra Python pairs | 20 | 84 | +64 pairs |

## Large-Scale D: Drive Audit (2026-05-06)

An audit of `workspace\comparisons` was performed to synchronize historical runs with the modern repository state.

### Findings
- The old `REVIEW_SUMMARY.md` on the D: drive is **obsolete** and references runs (e.g., `20260418_network_gate_trial`) that have been deleted.
- The `trace_order_fix` run is the current **source of truth** for edge candidate diagnostic work.
- Multiple abandoned intermediate runs (`180709_E_edges_a` through `l`) were identified as failed and have been purged.

### Cleanup Actions
- **Purged**: all intermediate `runs/180709_E_edges_*` to reclaim space.
- **Retained**: `trace_order_fix`, `180709_E_may2026_fixes`, and the canonical `180709_E_batch_190910-103039` oracle.
- **Audit Log**: A detailed summary is preserved at `workspace\comparisons\AUDIT_2026_05_06.md`.

### Key Findings

1. **Significant Improvement**: The trace order fix resulted in a 2.7x improvement in matched pairs (149 → 404) and 2.9x improvement in candidate generation (169 → 488).

2. **Match Rate**: Achieved 33.8% match rate (404/1,197 MATLAB pairs), up from 12.4% baseline.

3. **Remaining Gap**: Still missing 793 MATLAB pairs (66.2% of total), indicating substantial work remains.

4. **Extra Candidates**: Python now generates 84 pairs not present in MATLAB, suggesting potential over-generation or filtering differences.

5. **Top Missing Vertices**: Vertex 1350 (5 missing pairs), 229 (4 pairs), 92 (4 pairs), 29 (4 pairs), 469 (4 pairs).

6. **Top Extra Vertices**: Vertex 1127 (3 extra pairs), 914 (2 pairs), 41 (2 pairs), 72 (2 pairs).

### Sample Missing Pairs
[0,134], [0,529], [2,17], [2,329], [3,329], [5,888], [6,7], [6,9], [7,8], [12,31]

### Sample Extra Pairs
[15,351], [23,84], [24,445], [25,914], [28,1122], [33,34], [41,326], [41,410], [57,663], [62,142]

### Next Steps

1. **Investigate Missing Pairs (INVEST-001)**: Categorize the 793 missing MATLAB pairs to identify root causes (frontier ordering, join cleanup, sentinel lifecycle, etc.).

2. **Analyze Extra Pairs (INVEST-006)**: Understand why Python generates 84 pairs not in MATLAB (over-generation, incorrect filtering, frontier management differences).

3. **Baseline Clarification (PARITY-001A)**: Resolved discrepancy between claimed 41.4% baseline and actual measured rates. The 41.4% claim was determined to be inaccurate (likely a projection or target); the verified baseline progression is 12.4% → 33.8%.
4. **Investigation Completion (INVEST-001/006)**: Categorized 793 missing MATLAB pairs and 84 extra Python pairs. Identified top 3 root causes (Frontier Ordering, Hub Vertex Exploration, Filtering Gaps).

### Baseline History & Clarification

| Experiment | Date | Match Rate | Matched Pairs | Status |
|------------|------|------------|---------------|--------|
| TODO.md claim | Unknown | 41.4% | 496/1,197 | **Inaccurate** |
| may2026_fixes | 2026-05-04 | 12.4% | 149/1,197 | Verified Baseline |
| trace_order_fix | 2026-05-05 | 33.8% | 404/1,197 | Current State |

**Conclusion**: The actual progress is a 2.7x improvement in matched pairs (149 → 404) since the start of May 2026 fixes. The 41.4% figure should be disregarded as a historical measurement.

### Investigation Results (INVEST-001/006)

The investigation into the remaining gap identified the following priority root causes:

1. **Frontier Ordering Divergence** (High Impact): Systematic patterns in high-degree vertices (e.g., vertex 1350) suggest Python's frontier management differs from MATLAB.
2. **Hub Vertex Exploration**: Seed selection and frontier priority divergences for complex junctions.
3. **Over-Generation**: Python generates 84 pairs not in MATLAB, likely due to looser filtering or join cleanup timing differences.

### Next Proof Actions

## Exact Params Fairness Gate

The maintained exact route now rejects slavv_python runs whose saved
`validated_params.json` still carries Python-only parity controls or omits the
required MATLAB-shaped exact settings.

The fairness surface must include both serialized MATLAB settings and released
MATLAB slavv_python constants that are not written into `settings/*.mat`. The current
maintained exact bootstrap now records at least these source-level edge
constants explicitly:

- `step_size_per_origin_radius = 1`
- `max_edge_energy = 0`
- `edge_number_tolerance = 2`
- `distance_tolerance_per_origin_radius = 3`
- `energy_tolerance = 1`
- `radius_tolerance = 0.5`
- `direction_tolerance = 1`

Exact-route experiments should now also persist:

- `01_Params/shared_params.json`
- `01_Params/python_derived_params.json`
- `01_Params/param_diff.json`

Current live read on the preserved run root
`20260421_accepted_budget_trial`:

- `energy_projection_mode` is not explicitly recorded as `matlab`

Until that params surface is cleaned up, the run is not a fully fair
start-from-the-same-settings exact baseline even if later proof artifacts are
available.

## Native Energy

The maintained `hessian` path is now the canonical exact-compatible slavv_python for
energy generation.

Maintained native-energy coverage includes:

- projected `energy`
- `scale_indices`
- `energy_4d`
- per-scale Laplacian intermediates
- per-scale valid-mask behavior
- direct versus resumable alignment

This removes runtime dependence on imported MATLAB energy artifacts for the
canonical exact route.

## Vertices

Vertex extraction is source-aligned and downstream-ready on the native-first
exact route. No current evidence suggests that vertices are the first failing
surface.

## Edges: v22 Global Watershed

### Latest Maintained Candidate Snapshot

The last maintained v22 `capture-candidates` read remains:

| Metric | Count | vs MATLAB |
| --- | --- | --- |
| MATLAB candidates | 2533 | 100% oracle |
| Python candidates | 2120 | 83.7% |
| Matched pairs | 1643 | 64.9% match |
| Missing pairs | 890 | 35.1% gap |
| Extra pairs | 477 | 22.5% over |

**Note**: These counts are from before the May 2026 critical bug fixes. Re-run
`capture-candidates` to measure the actual improvement from the directional
suppression and trace order fixes.

### May 2026 Critical Bug Fixes

Two critical MATLAB parity bugs were identified and fixed on 2026-05-04:

#### 1. Directional Suppression in Seed Loop (VERIFIED)

**Status**: Verified Correct (2026-05-05)

**Finding**:
Previous reports (2026-05-04) suggested that directional suppression was outside the seed loop in MATLAB. However, exhaustive review of `get_edges_by_watershed.m` (line 763) confirms that directional suppression **IS** inside the `seed_idx` loop and is **iterative**.

**Impact**:
Python's current implementation of iterative suppression matches MATLAB's intent. The previous plan to move it outside the loop would have been a divergence.

**Action**:
Maintain the iterative suppression inside the seed loop in `slavv_python/core/_edge_candidates/global_watershed.py`.

#### 2. Trace Order Randomization (VERIFIED)

**Status**: Verified Correct (2026-05-05)

**Bug**: Python randomized trace point order only when `comparison_exact_network=True`, but MATLAB always uses `randperm`.
**Fix**: Always use seeded RNG for trace order shuffling in `slavv_python/core/edges_internal/edge_selection.py`. Matches MATLAB's `randperm` behavior on all routes.

#### 3. Distance Normalization (r/R) - ✅ FIXED (2026-05-05)

**Bug**: Python was using absolute micron distances for energy penalties. MATLAB uses relative distances normalized by the vessel radius ($R$) at each step.
**Fix**: Implemented `r/R` normalization in `common.py` and `global_watershed.py`. Penalties now correctly scale with vessel size.

#### 5. Backtracking Pointer Correction - ✅ FIXED (2026-05-05)

**Bug**: Watershed pointer indices were being generated sequentially, rather than indices that point back to the neighborhood origin. This broke the backtracking mechanism, leading to incomplete or incorrect edge traces.
**Fix**: Updated `_build_matlab_global_watershed_lut_cached` to correctly calculate reverse indices (where `subscripts[j] == -subscripts[i]`). Traces now correctly recover both origin vertices.

#### 6. Stable Discovery Edge Sorting - ✅ FIXED (2026-05-05)

**Bug**: Python discovered edges were processed in discovery order, but MATLAB explicitly sorts them by energy quality (max bottleneck) before subsequent stages. This caused downstream structural reconfiguration to diverge.
**Fix**: Implemented stable `np.argsort` by metric in `_finalize_matlab_parity_candidates`.

#### 9. Filtering Stage Order - ✅ FIXED (2026-05-05)

**Bug**: Python was performing `clean_edge_pairs` (best trajectory selection) before `crop_edges_V200` (out-of-bounds removal). This allowed an out-of-bounds edge with better energy to "block" a valid in-bounds edge during unique-pair selection, resulting in no edge being kept for that vertex pair.
**Fix**: Refactored `_choose_edges_matlab_style` to perform cropping as the very first step, passing only in-bounds candidates to the pair cleanup stage. This strictly matches the sequence found in `vectorize_V200.m`.

### Landed Fixes That Should Stay

The current exact-route watershed path has already absorbed these meaningful
fixes:

- ✅ aligned filtering order (Crop -> Pair Cleanup)
- ✅ disabled conflict painting (matched selection workflow)
- ✅ backtracking pointer correction (fixed trace recovery)
- ✅ stable discovery edge sorting (matched processing order)
- ✅ stable bridge vertex sorting (matched structural order)
- ✅ `r/R` distance normalization (scale-aware penalties)
- ✅ energy map integrity (stopped penalty leakage)
- ✅ iterative directional suppression in seed loop (iterative steering)
- ✅ enforced profile defaults (correctly applied `matlab_compat` settings)
- clipped-scale consistency between LUT creation and `size_map` storage
- MATLAB-style join-time reset behavior for `available_locations`
- MATLAB-aligned shared-state dtypes for `pointer_map` and `d_over_r_map`
- direct linear-offset backtracking for half-edge tracing
- final energy and scale sampling directly from the assembled MATLAB-order
  linear trace
- MATLAB-derived scale-tolerance calculation from the first two vessel radii

### Strongest Remaining Candidate Surfaces

After the May 2026 breakthroughs, the remaining candidate surfaces to investigate are:

1. Hub Vertex Exploration: Fine-grained divergences in complex junction geometries.
2. Boundary Conditions: Discrepancies at the extreme volume edges.
3. Filtering Gaps: Candidates missing from final output despite discovery (now largely addressed by conflict painting fix).

The recent series of fixes (especially pointers and conflict painting) is expected to result in a massive jump in match rate. Re-run parity experiments to confirm.

## Cleanup And Network

The cleanup chain is structurally aligned after the April 2026 audit:

- degree cleanup removal order matches MATLAB
- orphan cleanup terminal union matches MATLAB
- cycle cleanup removes the worst edge per component and prunes vertices in the
  same overall way

That means downstream proof is still blocked primarily by unresolved upstream
edge parity rather than known cleanup-specific bugs.

Network and strand assembly remain source-aligned but proof-pending until edge
parity closes.

## Historical Imported-MATLAB Replay Notes

These measurements came from the older imported-MATLAB replay track before the
native-first exact route became canonical. They are kept here only as historical
context.

### Historical First Exact Failure

- stage: `edges`
- field: `connections`
- MATLAB shape: `2533 x 2`
- Python shape: `1654 x 2`

### Historical Candidate Gap

Before the v22 route:

- raw Python candidates: `2364`
- intersection with MATLAB endpoint pairs: `2054`
- missing MATLAB pairs: `479`
- extra Python pairs: `310`

After the old chosen-edge path:

- final Python chosen edges: `1654`
- final chosen-edge intersection: `1553`
- final missing MATLAB pairs: `980`
- final extra Python pairs: `101`

### Historical Cleanup-Gate Fix

Removing the stale nonnegative-energy cleanup gate improved the historical
chosen-edge path from `1553` matched MATLAB pairs to `1886`, but it still did
not close parity.

## Next Proof Actions

1. **IMMEDIATE**: Re-run native-first `capture-candidates` to measure the actual
   improvement from the May 2026 directional suppression fix. This should
   dramatically improve or close the 16.3% candidate generation gap.
2. Re-run `prove-exact --stage edges` and record the first failing field.
3. If candidate generation now matches, investigate the 1 failing frontier
   ordering test as the next parity surface.
4. Clean the exact source-run params surface so preflight passes the fairness
   audit with MATLAB-shaped exact settings.
5. Keep `MATLAB_PARITY_MAPPING.md` focused on structural deviations and this
   file focused on live proof status.
6. Once edges pass, run `prove-exact --stage all` to close vertices and network.
