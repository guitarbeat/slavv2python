# MATLAB/Python Code Audit 2026-04-08

This file is the code-audit note for the April 8, 2026 parity investigation.

Read this file when you want:

- the side-by-side MATLAB vs Python control-flow comparison
- the concrete cleanup-path mismatches
- the official-doc checks for runtime semantics
- the best next implementation target from the audit

Companion reports:

- [Imported-MATLAB Parity Evidence 2026-04-08](parity_investigation_notes_2026-04-08.md)
- [Parity Report Index 2026-04-08](parity_report_index_2026-04-08.md)

The point of this audit was to stop guessing from local tracing symptoms and
instead answer:

1. What does the active MATLAB code actually do?
2. What does the Python parity code actually do?
3. Which low-level runtime assumptions are supported by official docs?
4. Where are the concrete mismatches?

## Key Findings

- The active MATLAB `vectorize_V200.m` edge-cleaning path uses
  `clean_edge_pairs`, `clean_edges_vertex_degree_excess`,
  `clean_edges_orphans`, and `clean_edges_cycles`.
- The Python parity path currently mixes `clean_edge_pairs`-style dedupe with
  `choose_edges_V200`-style conflict painting.
- That means Python is currently modeling the wrong MATLAB cleanup surface.
- Even if `choose_edges_V200` were the right target, Python still does not
  reproduce it literally because MATLAB iterates each edge in randomized voxel
  order with `randperm`.
- The doc-backed low-level assumptions mostly look sound, so the audit points
  toward code-structure mismatch rather than a hidden MATLAB/Python runtime
  misunderstanding.
- Cleanup mismatch is real, but cleanup alone is not the full parity blocker;
  the candidate pool is still wrong upstream too.

## Scope

MATLAB files inspected:

- `external/Vectorization-Public/source/get_edges_for_vertex.m`
- `external/Vectorization-Public/source/vectorize_V200.m`
- `external/Vectorization-Public/source/clean_edge_pairs.m`
- `external/Vectorization-Public/source/clean_edges_vertex_degree_excess.m`
- `external/Vectorization-Public/source/clean_edges_orphans.m`
- `external/Vectorization-Public/source/clean_edges_cycles.m`
- `external/Vectorization-Public/source/choose_edges_V200.m`
- `external/Vectorization-Public/source/get_edge_metric.m`

Python files inspected:

- `source/slavv/core/tracing.py`
- `source/slavv/core/pipeline.py`
- `source/slavv/evaluation/comparison.py`
- `docs/reference/MATLAB_MAPPING.md`

Reference parity run used while auditing:

- `comparisons/20260408_current_checkout_fresh`

## Fresh Parity Baseline

The fresh imported-MATLAB Python rerun on this checkout still lands at:

- MATLAB vertices: `1682`
- Python vertices: `1682`
- MATLAB edges: `1379`
- Python edges: `1425`
- MATLAB strands: `682`
- Python strands: `681`

So the current branch still fails full parity on edges and strands.

## What The Active MATLAB Path Actually Does

The highest-signal finding from this audit is that the active MATLAB V200 path
does **not** use `choose_edges_V200` in the main cleanup path.

In `vectorize_V200.m`, the active edge cleanup sequence is:

1. `clean_edge_pairs`
2. `clean_edges_vertex_degree_excess`
3. `clean_edges_orphans`
4. `clean_edges_cycles`

This is visible in the active code around:

- `clean_edge_pairs` call near line 3620
- `clean_edges_vertex_degree_excess` call near line 3664
- `clean_edges_orphans` call near line 3671
- `clean_edges_cycles` call near line 3682

The `choose_edges_V200` branch is present, but commented out in the same
section of `vectorize_V200.m`.

That matters because it changes what the Python parity cleanup should be trying
to emulate.

## What The Python Parity Path Actually Does

The Python parity path currently funnels candidates through
`_choose_edges_matlab_style()` in `source/slavv/core/tracing.py`.

That function currently does:

1. self-edge / dangling filtering
2. non-negative energy rejection
3. directed dedup
4. antiparallel dedup
5. conflict painting against a painted volume
6. degree pruning
7. orphan pruning
8. cycle pruning

So the Python cleanup currently mixes:

- `clean_edge_pairs`-style ordering and dedup
- plus `choose_edges_V200`-style conflict painting
- plus later MATLAB cleanup helpers

This is not the same as the active cleanup chain in `vectorize_V200.m`.

## Concrete Mismatch 1: Wrong MATLAB Cleanup Surface

This is the main structural mismatch found in the audit.

MATLAB V200 active path:

- `clean_edge_pairs`
- `clean_edges_vertex_degree_excess`
- `clean_edges_orphans`
- `clean_edges_cycles`

Python parity path:

- `_choose_edges_matlab_style()`
- which injects conflict painting before degree/orphan/cycle cleanup

Conclusion:

- Python is currently modeling a MATLAB cleanup helper that the active V200
  pipeline does not appear to use in the live edge-cleaning path.
- That is not a tiny tuning issue. It is a mismatch in the cleanup model.

## Concrete Mismatch 2: Even If We Wanted `choose_edges_V200`, Python Still Does Not Match It Literally

Inside MATLAB `choose_edges_V200.m`, each edge's positions are checked in a
random order:

- `edge_position_index_range = uint16(randperm(degrees_of_edges(edge_index)));`

Python conflict painting inside `_choose_edges_matlab_style()` walks the trace
in fixed stored order.

So even the MATLAB helper Python is borrowing from is not being reproduced
literally.

This does not prove `randperm` is the reason for the parity gap, but it does
prove that the current Python cleanup is not a literal port of
`choose_edges_V200` either.

## What `clean_edge_pairs.m` Actually Does

The active MATLAB dedupe helper:

1. computes edge metric with `get_edge_metric`
2. pre-sorts trajectories by length from shortest to longest
3. then sorts by mean/activation metric ascending
4. keeps the first unique directed pair with `unique(...,'rows','stable')`
5. removes the worse member of antiparallel pairs with `intersect(...,'rows','stable')`

Important detail:

- `get_edge_metric.m` is currently configured to use `max` energy, not mean.

This aligns with the Python parity metric helper:

- `_edge_metric_from_energy_trace()` uses `np.nanmax(...)`

So the metric choice itself does not look like the primary mismatch here.

## Official Runtime-Semantics Checks

The following assumptions used by the parity path are supported by official
docs.

### MATLAB sparse defaults

MathWorks documents that `spalloc(m,n,nz)` creates an all-zero sparse matrix:

- <https://www.mathworks.com/help/matlab/ref/spalloc.html>

This supports the Python assumption that unseen sparse-map entries in the
frontier logic behave like zero.

### MATLAB `min` tie behavior

MathWorks documents that `[M,I] = min(...)` returns the index of the **first**
occurrence of the minimum value:

- <https://www.mathworks.com/help/matlab/ref/double.min.html>

This supports the Python frontier tiebreak assumption that lowest energy then
lowest MATLAB-linear index is the right behavior when values tie.

### MATLAB linear indexing

MathWorks documents MATLAB linear indexing and `sub2ind` behavior here:

- <https://www.mathworks.com/help/matlab/math/detailed-rules-about-array-indexing.html>

This supports the Python parity code's explicit MATLAB-style linear index
conversion in `tracing.py`.

### MATLAB `unique`

MathWorks documents:

- `unique(A,setOrder)` with `'stable'`
- `occurrence` with `'first'` as the default

here:

- <https://www.mathworks.com/help/matlab/ref/double.unique.html>

This supports the Python parity logic that preserves first-seen directed pairs
after stable best-to-worst ordering.

### MATLAB `intersect`

MathWorks documents:

- `intersect(A,B,setOrder)`
- `'stable'`
- row-wise intersection behavior

here:

- <https://www.mathworks.com/help/matlab/ref/double.intersect.html>

This supports the MATLAB reading of the antiparallel-pair removal logic.

### Python heap and tuple ordering

Python docs for `heapq` explicitly discuss tuple heap entries and tie-breaking
concerns:

- <https://docs.python.org/3/library/heapq.html>

Python tutorial docs say sequence comparisons are lexicographic:

- <https://docs.python.org/3/tutorial/datastructures.html>

This supports the Python implementation detail that `(energy, linear_index)`
tuples in the heap break ties lexicographically.

### NumPy stable sort

NumPy documents stable sort behavior here:

- <https://numpy.org/doc/stable/reference/generated/numpy.sort.html>

This supports the Python use of stable ordering where we intentionally want
equal-key items to preserve previous order.

## What The Doc Check Suggests

The official-doc pass did **not** expose a major low-level language-runtime
mistake in the frontier assumptions.

The doc-backed parts look mostly sound:

- sparse default zero behavior
- first-occurrence semantics for `min`
- MATLAB linear indexing assumptions
- stable/first semantics for `unique`
- stable row semantics for `intersect`
- Python tuple lexicographic comparisons
- NumPy stable sorting

So the audit points back toward code-structure mismatch, not a hidden
MATLAB-vs-Python language-rule misunderstanding.

## Offline Cleanup Experiment

I tested an offline recomputation on the fresh candidate pool using a cleanup
sequence closer to the active MATLAB V200 path:

1. valid terminal filtering
2. non-negative energy rejection
3. directed dedup
4. antiparallel dedup
5. degree pruning
6. orphan pruning
7. cycle pruning

Result:

- current Python chosen-edge set: `1425`
- offline no-conflict-painting cleanup result: `1554`

Interpretation:

- removing the conflict-painting step does not magically recover MATLAB parity
- candidate generation is still wrong upstream
- but the current Python cleanup model is still mismatched with the active
  MATLAB cleanup path

So there are likely **two** active problems:

1. the candidate pool is wrong upstream
2. the cleanup model is not matching the active MATLAB V200 path

## Main Conclusion

The biggest concrete result from this audit is:

- the Python parity path is currently emulating the wrong MATLAB cleanup
  surface

Specifically:

- the active MATLAB V200 path uses `clean_edge_pairs` plus later cleanup
  helpers
- Python currently mixes in `choose_edges_V200`-style conflict painting

That is a structural mismatch and should be treated as a real parity bug, not
just a tuning choice.

At the same time:

- fixing cleanup alone will not solve parity
- the fresh candidate pool still deviates from MATLAB before cleanup starts

So the right overall picture is:

- cleanup mismatch is real
- upstream candidate-generation mismatch is still real
- both need to be addressed, but cleanup should stop modeling the wrong
  MATLAB helper first

## Best Next Step

The highest-value next implementation step is:

1. replace `_choose_edges_matlab_style()` with the actual active MATLAB V200
   cleanup chain
2. rerun the imported-MATLAB parity surface
3. measure the remaining mismatch
4. treat the residual as the true frontier-generation gap

That gives a cleaner target for the next round of parity work than continuing
to debug frontier behavior while the downstream cleanup model is still not the
same as the active MATLAB code path.

## Related Files

- `external/Vectorization-Public/source/vectorize_V200.m`
- `external/Vectorization-Public/source/clean_edge_pairs.m`
- `external/Vectorization-Public/source/clean_edges_vertex_degree_excess.m`
- `external/Vectorization-Public/source/clean_edges_orphans.m`
- `external/Vectorization-Public/source/clean_edges_cycles.m`
- `external/Vectorization-Public/source/choose_edges_V200.m`
- `source/slavv/core/tracing.py`
