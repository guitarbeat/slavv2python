# Exact Proof Findings

[Up: Reference Docs](../README.md)

**Last Updated**: 2026-04-27

This note tracks the maintained proof state for the native-first exact route.

For canonical claim boundaries and the distinction between source-level porting
and full Python implementation of the released SLAVV method, see
`MATLAB_METHOD_IMPLEMENTATION_PLAN.md`.

## Scope

- The canonical exact route is `comparison_exact_network=True` with
  `python_native_hessian` as the canonical exact-compatible energy provenance.
- Preserved MATLAB vectors remain the oracle artifacts for proof.
- `matlab_batch_hdf5` remains accepted only as a historical compatibility and
  replay surface.
- `100%` means artifact-level equality against preserved MATLAB vectors, not
  just matching counts.

## Quick Status Summary

| Component | Status | Proof State | Blocker |
|-----------|--------|-------------|---------|
| **Native Energy** | ✅ Complete | Canonical source | N/A |
| **Vertices** | ✅ Ready | Downstream-ready on native route | Awaiting edge proof |
| **Edges** | 🔄 Active Work | Source-aligned, proof pending | Candidate alignment (v22) |
| **Network** | ⏸️ Blocked | Source-aligned, proof pending | Upstream edge parity |

## Current Status (April 2026)

### Energy: Native Implementation Complete ✅

The maintained `hessian` path is now the canonical exact-compatible source for
energy generation. **This milestone removes the runtime dependency on imported
MATLAB energy artifacts.**

**Maintained native-energy coverage:**
- ✅ Projected `energy` field
- ✅ `scale_indices` field
- ✅ `energy_4d` full 4D energy tensor
- ✅ Per-scale Laplacian intermediates
- ✅ Per-scale valid-mask behavior
- ✅ Direct versus resumable alignment verified

**Impact**: The proof boundary has moved downstream. Native energy is no longer
the blocker for full Python implementation claims.

### Vertices: Downstream-Ready ✅

Vertex extraction is source-aligned and ready for proof on the native-first
exact route. The vertex stage now operates correctly from native Hessian energy
without requiring imported MATLAB artifacts.

**Status**: Proof pending, but not blocked by vertex-stage issues. Waiting for
downstream edge proof to complete before formal vertex proof run.

### Edges: High-Performance Parity Port (Active Work) 🔄

**Current Iteration**: v22 (as of April 2026)

**Major Accomplishments**:
1. ✅ **Resolved frontier propagation stagnation bug**
   - Implemented heapq-based O(log N) min-priority traversal
   - Added LIFO tie-breaking via insertion counter (matches MATLAB's `find(..., 'last')`)
   - Fixed energy tolerance thresholding: `(1.0 - energy_tolerance)`
   - Corrected vertex-ownership check timing to prevent premature frontier termination

2. ✅ **Architecture overhaul: "Flat-first" 1D Fortran design**
   - Eliminated 3D coordinate bottlenecks
   - Direct linear views for all map operations
   - Matches MATLAB's pointer-offset math exactly

3. ✅ **Verified candidate generation on canonical sample**
   - Frontier now propagates correctly across energy landscape
   - Zero-candidate stagnation bug is resolved
   - Candidates are being generated (proof of concept established)

**Current Focus**: Exact vertex-pair alignment
- Fine-tuning trace-back boundary conditions
- Closing the gap between Python candidate counts and MATLAB oracle pairs
- Running continuous `capture-candidates` verification

**Known Remaining Work**:
- Complete exact count alignment with MATLAB oracle
- Verify conflict-painting acceptance order in `conflict_painting.py`
- Re-check cleanup chain (crop, degree, orphan, cycle) after candidate alignment

### Network: Source-Aligned, Awaiting Upstream ⏸️

Network assembly and strand construction are source-aligned with MATLAB but
remain blocked on unresolved edge parity. No network-specific issues are known;
the blocker is purely upstream.

**Status**: Ready for proof once edge parity is established.

## Historical Quantified Findings (Pre-v22)

**Note**: These measurements came from the historical imported-MATLAB replay
track before the v22 frontier overhaul. They are retained for historical context
but should be replaced with fresh native-first measurements once v22 candidate
alignment is complete.

### Historical Edge Proof Failure (April 22, 2026)

**First artifact failure**: `edges.connections`

- stage: `edges`
- field: `connections`
- MATLAB shape: `2533 x 2`
- Python shape: `1654 x 2`

This established that the exact-parity gap was an edge-stage math problem, not a
proof-harness formatting issue.

### Historical Candidate Generation Gap

Raw candidate generation (before v22 fixes):

- raw Python edge candidates: `2364`
- raw candidate intersection with MATLAB endpoint pairs: `2054`
- raw candidate missing MATLAB pairs: `479`
- raw candidate extra Python pairs: `310`

After the full chosen-edge path:

- final Python chosen edges: `1654`
- final chosen-edge intersection with MATLAB endpoint pairs: `1553`
- final chosen-edge missing MATLAB pairs: `980`
- final chosen-edge extra Python pairs: `101`

### Historical Fix: Removed Stale Cleanup Gate

A legacy nonnegative-energy rejection in the chooser path was removed because
MATLAB's active deterministic path no longer uses it.

Measured improvement:

**Before the fix:**
- after prepare: `1861`
- after full cleanup: `1654`
- MATLAB pair intersection at final chosen edges: `1553`
- missing MATLAB pairs at final chosen edges: `980`

**After the fix:**
- after prepare: `2364`
- after full cleanup: `2044`
- MATLAB pair intersection at final chosen edges: `1886`
- missing MATLAB pairs at final chosen edges: `647`

This was a significant improvement but insufficient to close parity. The v22
frontier overhaul addresses the root cause of candidate generation gaps.

## Current Status

- Native energy cutover: complete enough to make native Hessian the canonical exact-compatible source surface.
- `source/core/_edge_candidates/global_watershed.py`: Successfully resolved the zero-candidate stagnation bug. The frontier now propagates correctly across the energy landscape using a **heapq-based O(log N) min-priority traversal** and corrected ownership-reveal sequencing. Current work is focusing on exact vertex-pair alignment on the native-first route.
- Global watershed join-time `available_locations` removal now follows MATLAB's
  indexed `intersect(...)` reset behavior instead of a looser value-based
  filter; quantified downstream edge impact still needs a fresh native-first
  rerun
- Global watershed shared-state map dtypes now match MATLAB for `pointer_map`
  (`uint64`) and `d_over_r_map` (`double`/`float64`); quantified downstream
  edge impact still needs a fresh native-first rerun
- Global watershed half-edge backtracking now follows MATLAB's direct
  `tracing_location - strel_linear_LUT_range{...}(pointer_map(...))` linear
  offset step, and final edge energy/scale traces are now sampled directly
  from the assembled MATLAB-order linear trace instead of being re-sampled
  through coordinate clipping.
- Edge candidate generation: high-performance parity port
  - **Logic Alignment**: 
    - Re-implemented the watershed frontier as a **min-heap (`heapq`)** to ensure energy-priority traversal.
    - Implemented **LIFO tie-breaking** via an insertion counter to match MATLAB's `find(..., 'last')` priority.
    - Corrected energy tolerance thresholding to use `(1.0 - energy_tolerance)`, allowing growth into valid high-energy regions.
    - Fixed the vertex-ownership check to sample `vertex_index_map` *before* revealing new strel claims, preventing the frontier from seeing its own previous reveal as an "already claimed" neighbor and stopping prematurely.
  - **Architecture**: Shifted to a **"flat-first"** design using 1D Fortran-ordered direct linear views for all map operations, matching MATLAB's pointer-offset math and eliminating 3D coordinate bottlenecks.
  - **Current Status**: Iteration **`v22`** is currently verified to generate candidates on the canonical sample, proving that the frontier propagation bug is resolved. Exact count alignment is the next target once the remaining trace-back boundary conditions are tuned.
- Vertex exact proof: downstream-ready on the native-first route
- Edge exact proof: A fresh continuous `capture-candidates` run is currently verifying the exactly matched endpoint gaps.
- Network exact proof: still blocked downstream on unresolved edge parity

## Next Proof Targets

1. Re-run `prove-exact` on the native-first route and record the first failing
   downstream field.
2. Keep candidate-boundary and replay-edge measurements current whenever edge
   math changes.
3. Replace the historical imported-MATLAB counts above with native-first rerun
   measurements once they are available.
