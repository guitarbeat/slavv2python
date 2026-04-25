# Exact Proof Findings

[Up: Reference Docs](../README.md)

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

## Native-First Track

### 1. The exact-route gate now accepts native Hessian as canonical

The maintained exact-route gate no longer depends on imported MATLAB energy.
The current gate is:

- `comparison_exact_network=True`
- exact-compatible energy provenance
- canonical provenance: `python_native_hessian`

This means the live Python exact route can now run from raw-image native energy
while still comparing against preserved MATLAB vectors as the oracle.

### 2. Native matched-filter energy is now the canonical source surface

The maintained `hessian` path is now the canonical exact-compatible source for
energy generation.

Maintained native-energy coverage now checks:

- projected `energy`
- `scale_indices`
- `energy_4d`
- per-scale Laplacian
- per-scale valid-mask behavior
- direct versus resumable alignment

That work moves the proof boundary downstream: the primary open parity work is
no longer the runtime dependency on MATLAB-produced energy artifacts.

### 3. The main remaining blocker is still downstream edge parity

The best quantified mismatch evidence is still the historical edge-proof data
captured before the native-first cutover. That historical evidence remains
useful because the downstream exact edge path is substantially the same code
surface now routed from native energy.

Current practical interpretation:

- energy provenance is no longer the main blocker
- edge candidate emission and chooser cleanup remain the likely first failing
  downstream surfaces
- network proof remains blocked behind unresolved edge parity

## Historical Quantified Findings

These measurements came from the historical imported-MATLAB replay track and are
retained here until a fresh native-first rerun replaces them with updated
numbers.

### 4. The first real artifact failure was `edges.connections`

On the April 22, 2026 exact-proof rerun:

- stage: `edges`
- field: `connections`
- MATLAB shape: `2533 x 2`
- Python shape: `1654 x 2`

That established that the remaining exact-parity gap was an edge-stage math
problem, not merely a proof-harness formatting issue.

### 5. The gap was split across candidate generation and chosen-edge cleanup

On that same run:

- raw Python edge candidates: `2364`
- raw candidate intersection with MATLAB endpoint pairs: `2054`
- raw candidate missing MATLAB pairs: `479`
- raw candidate extra Python pairs: `310`

After the full chosen-edge path:

- final Python chosen edges: `1654`
- final chosen-edge intersection with MATLAB endpoint pairs: `1553`
- final chosen-edge missing MATLAB pairs: `980`
- final chosen-edge extra Python pairs: `101`

### 6. A stale Python-only cleanup gate was removed

A legacy nonnegative-energy rejection in the chooser path was removed from the
exact route because MATLAB's active deterministic path no longer uses it.

Measured replay effect:

Before the fix:

- after prepare: `1861`
- after full cleanup: `1654`
- MATLAB pair intersection at final chosen edges: `1553`
- missing MATLAB pairs at final chosen edges: `980`

After the fix:

- after prepare: `2364`
- after full cleanup: `2044`
- MATLAB pair intersection at final chosen edges: `1886`
- missing MATLAB pairs at final chosen edges: `647`

That was a large improvement, but not enough to close parity.

## Current Status

- Native energy cutover: complete enough to make native Hessian the canonical
  exact-compatible source surface
- Global watershed size-tolerance derivation now matches the released MATLAB
  `get_edges_V300.m` first-two-radii formula; quantified downstream edge impact
  still needs a fresh native-first rerun
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
  - During a mid-run checkpoint, this logic successfully generated **2524 raw candidates**,
    a huge jump from the historical 2364 count, indicating this direct linear-offset method 
    recovers a major part of the 479 missing MATLAB pairs.
  - We discovered this exact mathematical port can occasionally form infinite cyclic pointers. 
    We injected a strict cycle-detector into `_matlab_global_watershed_trace_half` to gracefully 
    break them.
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
