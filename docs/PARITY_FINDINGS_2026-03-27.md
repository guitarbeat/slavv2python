# Parity Findings 2026-03-27

This note captures what the fresh March 27, 2026 MATLAB/Python reruns taught us
so the next parity phase can start from verified facts instead of chat history.

## Fresh Baselines

### Canonical parity rerun

- Run root: `comparison_output_live_parity`
- Source run root: `D:\slavv_comparisons\20260327_150656_clean_parity`
- MATLAB batch: `batch_260327-150756`
- Python energy source: `matlab_batch_hdf5`
- Python start point: imported MATLAB `energy` and `vertices`, rerun from `edges`

Observed parity status from `03_Analysis/comparison_report.json`:

- `vertices_exact = true`
- `edges_exact = false`
- `strands_exact = false`
- `passed = false`

Observed counts:

- MATLAB vertices: `1682`
- Python vertices: `1682`
- MATLAB edges: `1379`
- Python edges: `1236`
- MATLAB strands: `682`
- Python strands: `592`

Observed edge diagnostics:

- Candidate traces: `2150`
- Candidate endpoint pairs: `1493`
- MATLAB endpoint pairs: `1379`
- Matched MATLAB endpoint pairs in raw Python candidates: `889`
- Missing MATLAB endpoint pairs before cleanup: `490`
- Extra Python candidate endpoint pairs: `604`
- Conflict rejections: `244`
- Degree/orphan/cycle pruning: `5 / 3 / 5`
- First missing candidate endpoint pair sample: `[2, 356]`

Interpretation:

- Exact vertex parity is real and stable under imported MATLAB energy/vertices.
- The dominant blocker is still upstream of final cleanup because Python never
  generates many MATLAB endpoint pairs in the raw candidate set.
- Downstream cleanup still matters, but it cannot recover the missing `490`
  MATLAB endpoint pairs that never appear as candidates.

### Independent full Python rerun

- Run root: `D:\slavv_comparisons\20260327_161610_clean_python_full`
- MATLAB comparison target: the same fresh MATLAB batch above
- Python start point: native Python run from `energy`

Observed counts:

- Python vertices: `292`
- Python edges: `5`
- Python strands: `5`

Interpretation:

- Native-Python upstream generation on this volume is still far from MATLAB.
- The imported-energy parity path remains the correct surface for exact
  edge/strand work.
- Full native parity should not be conflated with the current exact-network
  parity effort.

## Operational Lessons

- Fresh MATLAB reruns failed repeatedly on `C:` with HDF5 write errors during
  the `energy` stage.
- The same command succeeded immediately on `D:` with ample free space.
- The failure mode was environmental, not a regression in the orchestration
  code or parameter file.

Implication:

- Future live MATLAB reruns should prefer a high-free-space local path such as
  `D:\slavv_comparisons\...` and only copy the finished parity run back into the
  repo when promotion is needed.

## Code-Path Lessons

### What is working

- `source/slavv/io/matlab_bridge.py` correctly imports MATLAB HDF5 energy and
  curated vertices into pipeline-compatible checkpoints.
- `comparison_exact_network=True` correctly routes MATLAB-energy parity runs
  through the parity-specific edge cleanup and network assembly path.
- Shared parity-aware network assembly in `source/slavv/core/graph.py` removed
  earlier fresh-vs-resume divergence.
- Candidate endpoint coverage diagnostics are useful and should remain a primary
  regression signal.

### What is not the dominant blocker

- Vertex matching is no longer the main issue on the parity path.
- Simple short-range terminal attachment tweaks did not improve live parity in
  earlier experiments and sometimes made it worse.
- Pure cleanup-order tuning is not enough by itself because too many MATLAB edge
  endpoint pairs are absent before cleanup starts.

### Important bug found during the fresh rerun

The independent Python-from-`energy` rerun initially crashed because MATLAB
settings were loaded as integer-like floats such as
`number_of_edges_per_vertex = 4.0`, and the tracer later used that value as an
array shape.

The validation layer now normalizes integral MATLAB-style scalars before the
pipeline uses them.

Implication:

- When replaying MATLAB batch settings into Python, assume MATLAB numeric values
  may arrive as floats even when the Python pipeline expects integers.

## Highest-Leverage Next Phase

### Primary target

Focus on upstream candidate generation in `source/slavv/core/tracing.py`,
especially where the parity tracer diverges from MATLAB before
`_choose_edges_matlab_style` runs.

The key question is:

- Why do only `889` of `1379` MATLAB endpoint pairs appear in the raw Python
  candidate set under imported MATLAB energy and exact vertices?

### Most promising direction

Port more of MATLAB's shared edge-discovery / watershed-join behavior rather
than continuing to tune local per-origin frontier termination heuristics.

Why:

- The current parity frontier is already good enough to hit exact vertices and a
  large subset of MATLAB edge pairs.
- The remaining failures look like missing cross-vertex/basin connectivity, not
  just local edge-choice mistakes.
- Earlier "near hit" or frontier-attachment experiments did not close the gap.

### Concrete next investigations

1. Compare MATLAB `get_edges_by_watershed.m` and the Python parity path for
   shared claim-map behavior, join-on-contact semantics, and how terminal
   ownership is resolved across origins.
2. Add a diagnostic artifact that records candidate endpoint pairs grouped by
   origin vertex so missing MATLAB pairs can be traced back to the exact origin
   vertices that never discover them.
3. For a few missing MATLAB endpoint pairs such as `[2, 356]`, inspect:
   - the MATLAB edge trace
   - the corresponding local energy neighborhood
   - whether Python launches from the same origin(s)
   - whether Python enters the same basin but fails to claim/merge it
4. Keep cleanup work gated to `comparison_exact_network=True` unless a change is
   clearly beneficial for default native-Python behavior too.

## Guardrails For The Next Pass

- Treat `comparison_output_live_parity` as the current trusted canonical parity
  artifact.
- Do not use old failed scratch runs on `C:` as evidence.
- Prefer fresh MATLAB reruns on `D:` and promote only after the full run
  completes cleanly.
- Use `candidate_endpoint_coverage` as the first regression screen before
  chasing exact final edge/strand diffs.
- Expect native Python-from-`energy` to remain non-parity for now; do not read
  that gap as proof that the imported-energy parity work regressed.
