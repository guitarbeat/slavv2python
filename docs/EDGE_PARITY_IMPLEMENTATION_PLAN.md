# Edge Parity Implementation Plan

Status: In Progress (Phase 2-3 diagnostics active; conflict-provenance reporting landed)
Date: 2026-04-06

## Context

The latest MATLAB-vs-Python parity reruns reached exact vertex parity but still
miss exact edge and strand parity.

Observed live rerun metrics on April 6, 2026:

- Vertices: `1682` MATLAB vs `1682` Python
- Edges: `1379` MATLAB vs `1425` Python
- Strands: `682` MATLAB vs `681` Python

Recent skip-MATLAB watershed-threshold experiments on the same imported-MATLAB
surface:

- `parity_watershed_metric_threshold = -90.0`
  - Edges: `1379` MATLAB vs `1387` Python
  - Strands: `682` MATLAB vs `654` Python
  - Final endpoint pairs: `893` matched / `486` MATLAB-only / `494` Python-only
- `parity_watershed_metric_threshold = -50.0`
  - Edges: `1379` MATLAB vs `1426` Python
  - Strands: `682` MATLAB vs `697` Python
  - Final endpoint pairs: `896` matched / `483` MATLAB-only / `530` Python-only
  - Candidate endpoint pairs: `2164` candidate / `990` matched / `389` missing

Repeatability check on April 6, 2026:

- Three fresh Python-only reruns from the same staged MATLAB batch produced the
  same Python result every time: `1425` edges, `681` strands, identical chosen
  endpoint-pair hashes, and identical chosen-trace hashes.

The current evidence still points to the edge-candidate path after the imported
MATLAB vertices stage, but the threshold experiments also showed that candidate
improvements can change final chosen-edge topology in non-obvious ways.

Conflict-provenance refresh on April 6, 2026
(`comparisons/20260406_conflict_provenance_refresh/03_Analysis/summary.txt`):

- Conflict rejects by source: `254` frontier, `741` watershed
- Conflict blockers by source: `868` frontier, `326` watershed
- Conflict source pairs:
  - `236` frontier -> frontier
  - `24` frontier -> watershed
  - `632` watershed -> frontier
  - `302` watershed -> watershed
- Chosen frontier edges:
  - `847` matched MATLAB, `391` extra
  - median energy `-225.4` matched vs `-152.3` extra
  - median trace length `11` matched vs `16` extra
- Chosen watershed edges:
  - `47` matched MATLAB, `140` extra
  - median energy `-118.6` matched vs `-75.3` extra
  - median trace length `17` matched vs `21` extra
- Strongest extra frontier edges overlapping missing MATLAB vertices:
  - `281/391` extra frontier edges share at least one vertex with a missing
    MATLAB endpoint pair
  - `18/20` of the strongest extra frontier edges share a vertex with at least
    one missing MATLAB endpoint pair
  - `41/50` of the strongest extra frontier edges do the same
- Shared-vertex missing-pair candidate hits from the same refresh:
  - vertex `359`: `0/4` missing MATLAB incident pairs ever appear as Python
    candidates
  - vertex `1283`: `0/4`
  - vertex `866`: `0/4`

Key references:

- `workspace/reports/python_matlab_parity_postfix_2026-03-30.md`
- `docs/PARITY_FINDINGS_2026-03-27.md`
- `source/slavv/core/tracing.py`
- `source/slavv/core/graph.py`

## Problem Statement

Python is producing too many candidate endpoint pairs and too many final edges
in the imported-MATLAB parity path. The extra edges then inflate the strand
count downstream, but blunt candidate pruning can also overshoot and destabilize
final strand assembly.

The most likely causes are:

1. The parity frontier tracer still diverges from MATLAB before cleanup.
2. The parity-only watershed supplement step is too permissive and adds extra
   pairs that MATLAB does not keep.
3. Cleanup logic in `_choose_edges_matlab_style()` can prune bad candidates, but
   it can also change topology in response to candidate-pool shifts, so better
   candidate coverage does not guarantee better final parity.

Current code-level findings from the April 6, 2026 live retest:

- The shorter-trace tie-break in Python is not a suspected divergence by
  itself; MATLAB's `clean_edge_pairs.m` pre-sorts by trajectory length before
  sorting by edge metric.
- Raw frontier candidates already contain a large extra set:
  - `892` matched MATLAB endpoint pairs
  - `615` frontier-only extra endpoint pairs
- Raw watershed candidates are even noisier:
  - `81` matched MATLAB endpoint pairs
  - `952` watershed-only extra endpoint pairs
- After cleanup, the final extra edge set is still dominated by frontier
  candidates:
  - chosen frontier edges: `847` matched / `391` extra
  - chosen watershed edges: `47` matched / `140` extra
- Extra frontier candidates are systematically longer and weaker than matched
  frontier candidates in the baseline run (median length `18` vs `12`, median
  metric `-156` vs `-223`).
- Provenance-aware conflict diagnostics now show that watershed candidates are
  often losing directly to already-painted frontier edges (`632`
  watershed->frontier conflict rejections), yet the final extra set still
  contains many more frontier edges than watershed edges.
- The strongest frontier extras usually touch the same vertices as missing
  MATLAB pairs, which makes the remaining gap look more like wrong partner
  selection around shared vertices than random frontier overgrowth.
- For the worst shared vertices inspected so far, the missing MATLAB incident
  pairs are absent from the Python candidate pool entirely. That shifts the
  current highest-confidence diagnosis upstream from cleanup into frontier
  candidate discovery.
- Some extra frontier edges touching those same vertices were generated from
  neighboring origins rather than from the shared vertex's own frontier search,
  so the next fix should inspect neighborhood-level frontier behavior and claim
  ordering, not just per-origin edge budgets.

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

Overly broad watershed supplementation is still a real source of extra endpoint
pairs, but global watershed metric thresholds are not sufficient on their own.
The recent threshold trials showed:

- stronger thresholds can reduce the edge-count gap,
- milder thresholds can improve candidate coverage,
- neither threshold produced better final edge and strand parity than the live
  no-threshold baseline.

That combination means the next fix probably needs to be selective rather than
global: preserve strand-critical watershed structure while reducing the
watershed candidates that only create extra Python topology. The new
conflict-provenance refresh sharpens that conclusion: watershed candidates
often lose to frontier blockers, but frontier survivors still dominate the
extra final edges. The frontier tracer and frontier claim ordering therefore
remain the highest-leverage surfaces, especially around shared vertices where
Python appears to be choosing the wrong partner endpoint.

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

- Prefer selective gates over a single global metric threshold.
- Limit supplements to pairs that are also consistent with origin-local
  topology or with conflict outcomes in `_choose_edges_matlab_style()`.
- Identify which rejected watershed candidates from the `-90.0` trial were
  strand-critical so the next gate preserves them.
- Compare chosen-edge set changes between the live retest and threshold trials
  before adding more pruning rules.
- Keep using the new provenance-aware conflict diagnostics to track when weak,
  long frontier winners displace MATLAB-like alternatives during conflict
  painting.

### Phase 3: Tighten frontier tracing semantics

Compare `_trace_origin_edges_matlab_frontier()` with the MATLAB
`get_edges_for_vertex.m` / `get_edges_by_watershed.m` behavior and align the
following areas:

- frontier ordering
- parent/child resolution
- frontier pruning beyond already-found terminal directions
- terminal hit handling and trace finalization
- why some high-confidence but still non-MATLAB frontier candidates survive
  cleanup and whether they correspond to nearby missing MATLAB endpoint pairs
- whether shared-vertex local ordering is selecting the wrong partner edge at
  vertices that appear in both the missing-MATLAB and extra-frontier sets
- which rule inside `_trace_origin_edges_matlab_frontier()` is preventing the
  missing MATLAB incident pairs from being generated at the top shared vertices

### Phase 4: Preserve cleanup as a downstream safety net

Keep `_choose_edges_matlab_style()` focused on dedupe and pruning, not on
compensating for upstream semantic drift. However, the threshold experiments now
show that cleanup order is part of the parity story, so conflict ordering and
source preference may need targeted diagnostics before the next edge-candidate
change. The next cleanup investigation should use the landed provenance-aware
reporting rather than another global threshold sweep.

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
- `tests/unit/core/test_candidate_diagnostics.py`
- `tests/unit/core/test_edge_cases.py`
- `tests/integration/test_regression_edges.py`

## Risks

- Over-constraining supplements may reduce coverage and regress edges that are
  currently correct.
- Frontier tracing changes can be subtle and may alter runtime or ordering.
- Graph-layer changes are unlikely to fix the root issue and should be avoided
  unless new evidence appears.

## Next Decision Point

If a selective watershed gate can reduce extra topology without repeating the
strand regressions from the `-90.0` and `-50.0` trials, keep iterating there
first.

If the strongest extra frontier edges still survive with clearly worse
energy/length profiles than matched frontier edges, the next patch should
inspect `_trace_origin_edges_matlab_frontier()` and local claim ordering before
making another watershed-filtering change.
