# Exact Proof Findings

[Up: Reference Docs](../README.md)

This note tracks the current imported-MATLAB exact-route proof findings while
artifact parity is still open.

For canonical claim boundaries and the distinction between source-level porting
and full Python implementation of the released SLAVV method, see
`MATLAB_METHOD_IMPLEMENTATION_PLAN.md`.

## Scope

- This document is only about the imported-MATLAB exact route.
- `100%` means artifact-level equality against preserved MATLAB vectors, not
  just matching counts.
- The proof source is the preserved MATLAB batch under the staged comparison
  run root, not `comparison_report.json`.

## Canonical Runs

- Preserved MATLAB source run:
  `D:\slavv_comparisons\experiments\live-parity\runs\20260421_accepted_budget_trial`
- First exact-proof rerun:
  `D:\slavv_comparisons\experiments\live-parity\runs\20260422_exact_proof_trial`

## Findings Through April 23, 2026

### 1. The proof contract needed correction before the math could be judged

The first proof pass exposed representation mismatches that were not real math
failures:

- exact proof must prefer `curated_vertices*.mat` over raw `vertices*.mat`
- MATLAB spatial coordinates must be reversed into the Python checkpoint axis
  order before comparison
- the destination exact-route vertex checkpoint must be synced from the
  canonical MATLAB vertex vectors before rerunning downstream stages

After those fixes, the vertex stage passed exact proof.

### 2. The first real artifact failure is in `edges.connections`

After the proof-surface fixes, the first true mismatch in
`20260422_exact_proof_trial` was:

- stage: `edges`
- field: `connections`
- MATLAB shape: `2533 x 2`
- Python shape: `1654 x 2`

That established that the remaining exact-parity gap was still a real edge-stage
math problem, not a proof-harness formatting issue.

### 3. The gap is split across both candidate generation and chosen-edge cleanup

On the April 22, 2026 exact-proof rerun:

- raw Python edge candidates: `2364`
- raw candidate intersection with MATLAB endpoint pairs: `2054`
- raw candidate missing MATLAB pairs: `479`
- raw candidate extra Python pairs: `310`

After the full chosen-edge path:

- final Python chosen edges: `1654`
- final chosen-edge intersection with MATLAB endpoint pairs: `1553`
- final chosen-edge missing MATLAB pairs: `980`
- final chosen-edge extra Python pairs: `101`

So the remaining failure is not only in candidate generation. The exact route is
also losing many MATLAB-valid pairs later in edge cleanup and selection.

### 4. A stale Python-only energy gate was still active in `clean_edge_pairs`

The most important April 23, 2026 finding was that Python was still applying a
legacy rejection that MATLAB's active deterministic path no longer uses.

Before the fix, `prepare_candidate_indices_for_cleanup(...)` in
`source/slavv/core/_edge_selection/payloads.py` removed any candidate whose
energy trace ever reached `>= 0`.

That rule is not part of active MATLAB `clean_edge_pairs.m` for the
deterministic search path. In the MATLAB source, the old nonnegative-energy
filter is commented out.

Measured effect on the exact candidate surface from
`20260422_exact_proof_trial`:

- raw candidates entering cleanup: `2364`
- candidates after the stale Python energy gate: `1861`
- candidates removed only by that gate: `503`
- MATLAB-valid pairs removed by that gate: `397`

This was a real mathematical deviation, not just a bookkeeping difference.

### 5. The exact route now bypasses that stale gate

The imported-MATLAB exact route now disables the nonnegative-energy rejection,
while legacy non-parity callers keep the old behavior.

Changed surfaces:

- `source/slavv/core/_edge_selection/payloads.py`
- `source/slavv/core/_edge_selection/conflict_painting.py`
- `source/slavv/core/_edges/bridge_vertices.py`

This keeps the exact route aligned with MATLAB's active deterministic
`clean_edge_pairs.m` semantics without widening that behavior across unrelated
legacy paths.

### 6. The replayed effect is large, but parity is still not closed

Replaying the exact chooser locally on the same April 22, 2026 candidate surface
showed a large improvement after removing the stale gate:

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

So this one fix closes a large part of the exact-route edge gap, but it does
not yet achieve artifact equality with MATLAB.

### 7. The remaining exact gap is now smaller and more focused

After removing the stale energy gate, the remaining open issues are:

- candidate generation still misses `479` MATLAB pairs before cleanup
- later edge selection and cleanup still remove many MATLAB-valid pairs even
  after the stale energy gate is gone

The next audit targets are therefore:

1. exact global watershed candidate emission
2. conflict-painting acceptance order
3. crop / degree / cycle cleanup behavior on the exact route

## Current Status

- Vertex exact proof: passing after proof-contract fixes
- Edge exact proof: failing, but materially closer after the April 23, 2026
  `clean_edge_pairs` fix
- Network exact proof: still blocked downstream on the unresolved edge mismatch

## Claim Boundary

The exact imported-MATLAB route is not yet proven `100%` equal to MATLAB.
The honest status is:

- many core mathematical surfaces are now source-level aligned
- at least one major Python-only deviation has been removed
- full artifact proof is still red until `vertices`, `edges`, and `network`
  all pass `prove-exact`
