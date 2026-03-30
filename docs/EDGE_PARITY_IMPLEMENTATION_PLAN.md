# Edge Parity Implementation Plan

Status: Draft
Date: 2026-03-30

## Context

The latest MATLAB-vs-Python parity rerun reached exact vertex parity but still
misses exact edge and strand parity.

Observed full-run metrics:

- Vertices: `1682` MATLAB vs `1682` Python
- Edges: `1379` MATLAB vs `1560` Python
- Strands: `682` MATLAB vs `820` Python

The current evidence points to the edge-candidate path after the imported
MATLAB vertices stage, not to vertex extraction or graph assembly.

Key references:

- `workspace/reports/python_matlab_parity_postfix_2026-03-30.md`
- `docs/PARITY_FINDINGS_2026-03-27.md`
- `source/slavv/core/tracing.py`
- `source/slavv/core/graph.py`

## Problem Statement

Python is producing too many candidate endpoint pairs and too many final edges
in the imported-MATLAB parity path. The extra edges then inflate the strand
count downstream.

The most likely causes are:

1. The parity frontier tracer still diverges from MATLAB before cleanup.
2. The parity-only watershed supplement step is too permissive and adds extra
   pairs that MATLAB does not keep.
3. Cleanup logic in `_choose_edges_matlab_style()` can prune bad candidates, but
   it cannot restore missing MATLAB endpoint pairs or remove all semantic
   mismatches introduced upstream.

## Goals

- Reduce Python candidate endpoint pairs so they align with MATLAB.
- Preserve exact vertex parity.
- Keep the parity path deterministic and repeatable.
- Restore exact edge and strand parity on the canonical imported-MATLAB run.

## Non-Goals

- Do not change native Python-from-TIFF behavior unless a fix also benefits the
  parity path.
- Do not rework graph assembly as the primary fix.
- Do not broaden cleanup heuristics just to mask upstream mismatch.

## Current Code Path

The parity flow currently looks like this:

1. `comparison.py` enables `comparison_exact_network=True` for parity runs.
2. `extract_edges()` in `source/slavv/core/tracing.py` selects the MATLAB
   frontier tracer when `energy_origin == "matlab_batch_hdf5"`.
3. `_trace_origin_edges_matlab_frontier()` produces per-origin candidate traces.
4. `_supplement_matlab_frontier_candidates_with_watershed_joins()` adds
   watershed-touching pairs that the frontier tracer missed.
5. `_choose_edges_matlab_style()` dedupes and prunes candidates.
6. `source/slavv/core/graph.py` turns the chosen edges into strands.

## Working Hypothesis

The extra 181 Python edges are primarily caused by overly broad watershed
supplementation, while some true MATLAB pairs are still missing from the raw
frontier candidate set.

That combination matches the latest report:

- raw frontier candidates still miss MATLAB endpoint pairs
- watershed supplementation contributes a large number of additional pairs
- the final edge count remains above MATLAB

## Proposed Implementation Plan

### Phase 1: Measure the candidate gap precisely

Add or expand diagnostics so we can answer these questions for the parity run:

- Which MATLAB endpoint pairs are missing before supplementation?
- Which candidate pairs are added only by the watershed supplement step?
- Which seed origins are responsible for the missing pairs?
- Which origin vertices generate the largest number of extra candidate pairs?

Recommended code touch points:

- `source/slavv/evaluation/metrics.py`
- `source/slavv/core/tracing.py`

### Phase 2: Constrain watershed supplementation

Audit `_supplement_matlab_frontier_candidates_with_watershed_joins()` and its
helpers so that the supplement step only fills true MATLAB-like gaps.

Candidate changes to evaluate:

- Require stricter contact validation before adding a pair.
- Limit supplements to pairs that are also consistent with frontier reachability
  or origin-local topology.
- Add a parity-only gating rule so the supplement step can be compared against a
  known MATLAB endpoint set before it is accepted.

### Phase 3: Tighten frontier tracing semantics

Compare `_trace_origin_edges_matlab_frontier()` with the MATLAB
`get_edges_for_vertex.m` / `get_edges_by_watershed.m` behavior and align the
following areas:

- frontier ordering
- parent/child resolution
- frontier pruning beyond already-found terminal directions
- terminal hit handling and trace finalization

### Phase 4: Preserve cleanup as a downstream safety net

Keep `_choose_edges_matlab_style()` focused on dedupe and pruning, not on
compensating for upstream semantic drift.

## Acceptance Criteria

The fix is considered successful when the canonical parity rerun produces:

- exact vertex match
- exact edge match
- exact strand match
- stable rerun-to-rerun candidate counts
- no regression in the imported-MATLAB parity workflow runtime

## Verification Plan

Run the targeted parity checks first, then the full comparison:

1. Re-run the diagnostic comparison that reports candidate endpoint coverage.
2. Verify the missing MATLAB endpoint pair count decreases.
3. Verify the extra candidate endpoint pair count decreases.
4. Re-run the full parity workflow and confirm the final edge and strand counts
   match MATLAB exactly.

Helpful references:

- `tests/unit/analysis/test_comparison_metrics.py`
- `tests/unit/core/test_edge_cases.py`
- `tests/integration/test_regression_edges.py`

## Risks

- Over-constraining supplements may reduce coverage and regress edges that are
  currently correct.
- Frontier tracing changes can be subtle and may alter runtime or ordering.
- Graph-layer changes are unlikely to fix the root issue and should be avoided
  unless new evidence appears.

## Next Decision Point

If candidate-coverage diagnostics show the supplement step is responsible for
most extra edges, the next patch should focus there first.

If the supplement step looks reasonable, the frontier tracer itself should be
compared line-by-line against MATLAB behavior next.
