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
| **Edges** | 🔧 Fixing | v22 bugs being addressed | Fixes applied, testing in progress |
| **Network** | ⏸️ Blocked | Source-aligned, proof pending | Upstream edge parity |

## Current Status (April 2026)

### Critical Finding (April 27, 2026): v22 Candidate Generation Bugs — Fixes In Progress

**Initial Run** (April 27, 2026 morning): `capture-candidates` FAILED with critical errors:
- 40+ cycle detection errors in backtracking
- 15+ pointer index out-of-range errors

**Fixes Applied** (April 27, 2026 afternoon):
1. Added bounds checking after computing next backtrack location
2. Added pointer validation in reveal function to filter invalid pointers
3. Added LUT consistency checks and assertions

**Current Status** (April 27, 2026 1:30 PM): Testing in progress
- `capture-candidates` is running with fixes applied
- No immediate crashes observed
- Algorithm is taking longer than expected (5+ minutes, still running)
- Awaiting completion to assess if valid candidates are generated

See `V22_BUG_FIXES.md` for detailed fix descriptions.

**Impact**: The immediate crashes are prevented by defensive checks, but the
algorithm performance and correctness still need verification. The long runtime
suggests either the defensive filtering is catching many invalid pointers, or
there are additional performance/logic issues.

**Immediate Next Steps**:
1. Wait for current test run to complete
2. Examine output for valid candidate generation
3. Check logs for how many invalid pointers were filtered
4. Add iteration limits if needed to prevent infinite loops

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

### Edges: v22 Blocked on Critical Bugs ❌

**Current Iteration**: v22 (April 2026)

**Status as of April 27, 2026**: **BLOCKED** — `capture-candidates` run revealed
critical bugs in the v22 global watershed implementation that prevent valid
candidate generation.

**Previous Accomplishments** (now invalidated):
1. ✅ **Resolved frontier propagation stagnation bug**
   - Implemented heapq-based O(log N) min-priority traversal
   - Added LIFO tie-breaking via insertion counter (matches MATLAB's `find(..., 'last')`)
   - Fixed energy tolerance thresholding: `(1.0 - energy_tolerance)`
   - Corrected vertex-ownership check timing to prevent premature frontier termination

2. ✅ **Architecture overhaul: "Flat-first" 1D Fortran design**
   - Eliminated 3D coordinate bottlenecks
   - Direct linear views for all map operations
   - Matches MATLAB's pointer-offset math exactly

3. ❌ **"Verified candidate generation on canonical sample"** — INVALIDATED
   - Candidates are generated, but contain fundamental errors
   - Cycle detection errors during backtracking (40+ instances)
   - Pointer index out-of-range errors (15+ instances)
   - The "proof of concept" claim was premature

**Critical Bugs Found (April 27, 2026)**:

1. **Cycle Detection in Backtracking**:
   - 40+ instances of `Cycle detected in global watershed backtrack at <location>`
   - The pointer map contains cycles that prevent valid trace reconstruction
   - Likely caused by incorrect pointer writes during frontier propagation

2. **Pointer Index Out-of-Range**:
   - 15+ instances of `Pointer index <N> out of range for scale <S> (size <M>)`
   - Pointers exceed the LUT size for the given scale
   - Indicates the frontier is writing invalid pointer values

**Root Cause Hypothesis**:
- The flat-first 1D architecture or heapq traversal is writing pointers that
  don't respect LUT bounds or create circular references
- The LIFO tie-breaking or energy-tolerance logic may be causing the frontier
  to revisit locations in a way that creates invalid pointer chains

**Immediate Fix Requirements**:
1. Add defensive logging to capture pointer map state when errors occur
2. Verify pointer writes respect LUT size bounds for each scale
3. Check if tie-breaking or energy-tolerance logic creates cycles
4. Run small-scale debug trace to isolate the first failing candidate

**Known Remaining Work** (blocked until bugs are fixed):
- Fix cycle detection and pointer out-of-range bugs
- Complete exact count alignment with MATLAB oracle
- **Fix randomised trace-order in conflict-painting loop** — MATLAB uses `randperm`
  per edge; Python iterates sequentially. This is the only structural deviation
  found in the chooser after the April 2026 audit. See `MATLAB_PARITY_MAPPING.md`
  Deviation #3 for the full analysis and recommended fix.
- Re-check cleanup chain (crop, degree, orphan, cycle) after candidate alignment

### Network: Source-Aligned, Awaiting Upstream ⏸️

Network assembly and strand construction are source-aligned with MATLAB but
remain blocked on unresolved edge parity. No network-specific issues are known;
the blocker is purely upstream.

**Status**: Ready for proof once edge parity is established.

### Cleanup Chain: Structurally Aligned ✅

An April 2026 audit of `cleanup.py` against the three MATLAB cleanup functions
found all three structurally aligned:

- `clean_edges_vertex_degree_excess`: removal order (highest-index / worst-energy
  first) matches MATLAB exactly.
- `clean_edges_orphans`: terminal-location union (interior edge locations ∪ vertex
  locations) matches the active MATLAB path.
- `clean_edges_cycles`: worst-edge-per-component removal and between-iteration
  vertex pruning both match MATLAB.

No cleanup-specific fixes are needed. The cleanup chain will be re-verified once
upstream candidate alignment improves.

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
