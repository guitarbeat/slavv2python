# MATLAB/Python Code Audit 2026-04-08

## What this file is for

This is the canonical technical appendix for the April 8 parity diagnosis. It
consolidates the strongest code-path and artifact-backed evidence behind the
current conclusion.

## Read this when

- you want the strongest MATLAB vs Python cleanup mismatch evidence
- you need the fresh rerun numbers and candidate-coverage evidence
- you want to know which language/runtime assumptions were checked and cleared
- you need the implementation consequence of the audit, not just the symptom

## Executive Summary

- The active MATLAB V200 cleanup path uses `clean_edge_pairs`,
  `clean_edges_vertex_degree_excess`, `clean_edges_orphans`, and
  `clean_edges_cycles`.
- The Python parity path still routes candidates through
  `_choose_edges_matlab_style()`, which mixes in `choose_edges_V200`-style
  conflict painting before later cleanup.
- The fresh April 8 imported-MATLAB rerun still fails at MATLAB edges `1379`
  vs Python edges `1425`, and MATLAB strands `682` vs Python strands `681`.
- Artifact checks show the saved `chosen_edges.pkl` is internally consistent
  when recomputed with `validated_params.json`, so the fresh run evidence is
  trustworthy.
- Official-doc checks for sparse defaults, `min`, linear indexing, `unique`,
  `intersect`, Python tuple ordering, and NumPy stable sort do not point to a
  hidden language-runtime misunderstanding.
- The audit supports a two-part diagnosis: upstream candidate generation is
  still wrong, and downstream cleanup is still modeled against the wrong MATLAB
  surface.

## Handling Classification

- Status: Handled
- Why: The audit scope was to establish and defend the cleanup-chain mismatch
  diagnosis with artifact-backed evidence. That scope is complete, and the
  cleanup-chain alignment recommendation has been implemented in code.
- Remaining open work: Residual parity mismatch after cleanup alignment remains
  tracked as active algorithmic work in
  `docs/reports/handled/parity_decision_memo_2026-04-08.md`.

## Current Status

### Confirmed and stable

- Vertices are exact on the imported-MATLAB parity surface.
- Stage-isolated `network` parity is exact when exact MATLAB `edges` are
  imported and Python reruns from `network` with `comparison_exact_network`
  enabled.
- The fresh April 8 chosen-edge artifact is trustworthy when replayed with the
  effective `validated_params.json`.

### Active mismatch

- The fresh rerun still fails on edges and strands.
- Python still misses many MATLAB endpoint pairs before cleanup begins.
- Python cleanup still injects conflict-painting behavior that the active
  MATLAB V200 path does not use in its live cleanup chain.

### Superseded concerns

- Generic network assembly is not the main suspect.
- A hidden MATLAB-vs-Python runtime semantics mismatch is not the best current
  explanation.
- Cleanup-only explanations are incomplete because the candidate pool is still
  wrong upstream.

## Strongest Evidence

### Active MATLAB cleanup chain

In `vectorize_V200.m`, the active edge cleanup sequence is:

1. `clean_edge_pairs`
2. `clean_edges_vertex_degree_excess`
3. `clean_edges_orphans`
4. `clean_edges_cycles`

The `choose_edges_V200` branch exists, but it is not the active cleanup path
used by the V200 workflow section that matters here.

### Current Python parity cleanup chain

The Python parity path currently routes candidates through
`_choose_edges_matlab_style()` in `source/slavv/core/tracing.py`. That flow
currently combines:

1. self-edge and dangling filtering
2. non-negative energy rejection
3. directed dedup
4. antiparallel dedup
5. conflict painting against a painted volume
6. degree pruning
7. orphan pruning
8. cycle pruning

That is not the same as the active MATLAB cleanup chain. Even if
`choose_edges_V200` were the intended target, Python still does not reproduce
it literally because MATLAB iterates edge positions in randomized voxel order
with `randperm`.

### Fresh April 8 artifact evidence

The fresh imported-MATLAB rerun on `comparisons/20260408_current_checkout_fresh`
landed at:

- vertices: MATLAB `1682`, Python `1682`
- edges: MATLAB `1379`, Python `1425`
- strands: MATLAB `682`, Python `681`

Candidate coverage from `summary.txt` showed:

- candidate endpoint pairs: `2540`
- matched MATLAB candidate pairs: `973`
- missing MATLAB candidate pairs: `406`
- final matched pairs: `894`
- MATLAB-only final pairs: `485`
- Python-only final pairs: `531`

That means the candidate pool is already wrong before cleanup starts. Shared
vertex spot checks around `359`, `572`, `866`, and `1283` reinforce the same
story: active local neighborhoods are present, but the expected MATLAB partner
set never fully enters the Python candidate pool.

### Artifact trust check

An important consistency issue was resolved during the investigation:

- offline replay of `_choose_edges_matlab_style(...)` only matched the saved
  `chosen_edges.pkl` once the run's effective `validated_params.json` was used
- `comparison_params.normalized.json` alone was not sufficient because
  imported-MATLAB settings are overlaid after that snapshot is written

Conclusion:

- the saved chosen-edge artifact is trustworthy
- the mismatch is in the candidate/cleanup behavior, not in a corrupted saved
  chooser artifact

## Runtime semantics checks that still matter

The following assumptions were checked against official docs and remain
reasonable:

- `spalloc` sparse defaults behave like zero-filled sparse storage
- MATLAB `min` returns the first occurrence on ties
- MATLAB linear indexing assumptions used by the parity path are sound
- `unique(..., 'rows', 'stable')` and `intersect(..., 'rows', 'stable')`
  semantics align with the parity implementation strategy
- Python tuple ordering and NumPy stable sort behavior support the explicit
  tie-breaking choices used in the parity code

These checks are still useful guardrails, but they are not the main blocker.

## Implementation consequence

The highest-value consequence of this audit is straightforward:

- Python cleanup should stop modeling the wrong MATLAB helper surface
- the cleanup path should be aligned with the active MATLAB V200 chain first
- the next rerun should then be interpreted as the true residual upstream
  candidate/frontier gap

An offline cleanup experiment without conflict painting moved the chosen-edge
result from `1425` to `1554`, which confirms that cleanup modeling matters but
does not magically solve parity on its own.

## Recommended next actions

1. Replace `_choose_edges_matlab_style()` with a cleanup flow aligned to the
   active MATLAB V200 chain.
2. Rerun the imported-MATLAB parity surface immediately after that cleanup
   change.
3. Use the remaining mismatch as the true upstream frontier/candidate
   generation problem.
4. Keep `validated_params.json` as the authoritative replay input for any
   future offline artifact checks.
