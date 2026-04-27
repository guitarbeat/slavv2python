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

## Global Watershed Detail (v22)

These are the specific alignment fixes landed in the v22 iteration, recorded
here for audit traceability.

- `available_locations` removal at join-time now follows MATLAB's indexed
  `intersect(...)` reset behavior instead of a looser value-based filter.
- Shared-state map dtypes now match MATLAB: `pointer_map` is `uint64`,
  `d_over_r_map` is `float64`.
- Half-edge backtracking now follows MATLAB's direct
  `tracing_location - strel_linear_LUT_range{...}(pointer_map(...))` linear
  offset step.
- Final edge energy and scale traces are sampled directly from the assembled
  MATLAB-order linear trace instead of being re-sampled through coordinate
  clipping.

Quantified downstream impact of these fixes still needs a fresh native-first
`capture-candidates` rerun to replace the pre-v22 historical numbers above.

## Next Proof Targets

1. Run `capture-candidates` on the native-first route to get updated candidate
   counts against the MATLAB oracle and replace the pre-v22 historical numbers.
2. Run `prove-exact --stage edges` and record the first failing field and
   measured counts.
3. Keep candidate-boundary and `replay-edges` measurements current whenever
   edge math changes.
4. Once edges pass, run `prove-exact --stage all` to close vertices and network.
