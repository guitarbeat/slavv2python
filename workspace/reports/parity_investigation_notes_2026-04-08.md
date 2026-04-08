# Imported-MATLAB Parity Evidence 2026-04-08

This file is the evidence note for the fresh imported-MATLAB parity rerun on
the April 8, 2026 checkout.

Read this file when you want:

- the fresh parity baseline
- the current mismatch shape
- the shared-vertex spot checks from the latest rerun
- the current working diagnosis from artifacts

Companion reports:

- [MATLAB/Python Code Audit 2026-04-08](matlab_python_code_audit_2026-04-08.md)
- [Parity Report Index 2026-04-08](parity_report_index_2026-04-08.md)

## Key Findings

- Fresh rerun still fails full parity:
  - MATLAB edges `1379` vs Python edges `1425`
  - MATLAB strands `682` vs Python strands `681`
- The fresh run confirms the mismatch is real on the current checkout and not
  just a stale artifact.
- The saved `chosen_edges.pkl` is internally consistent when recomputed with
  `validated_params.json`.
- The candidate pool is still missing many MATLAB endpoint pairs before
  cleanup.
- Shared vertices like `359`, `572`, `866`, and `1283` still show local
  partner substitution or missing MATLAB-style partner generation.

## Fresh Baseline

- Fresh Python-only rerun root:
  `comparisons/20260408_current_checkout_fresh`
- MATLAB batch reused from:
  `comparisons/20260406_live_parity_retest/01_Input/matlab_results/batch_260406-164522`
- Command used:

```powershell
.\.venv\Scripts\python.exe workspace/scripts/cli/compare_matlab_python.py `
  --input data/slavv_test_volume.tif `
  --skip-matlab `
  --output-dir comparisons\20260408_current_checkout_fresh
```

- Result:
  - MATLAB vertices: `1682`
  - Python vertices: `1682`
  - MATLAB edges: `1379`
  - Python edges: `1425`
  - MATLAB strands: `682`
  - Python strands: `681`
- Status:
  - vertices `PASS`
  - edges `FAIL`
  - strands `FAIL`

This fresh rebuild reproduces the same `+46` edge / `-1` strand mismatch shape
seen in the earlier April 6 parity artifacts. The gap is real on the current
branch and is not only a stale-run artifact.

## Runtime Consistency Finding

One internal consistency issue was resolved during the investigation:

- Recomputing `_choose_edges_matlab_style(...)` offline from the fresh
  `candidates.pkl` initially produced a different answer until the runtime's
  actual `validated_params.json` was used.
- `comparison_params.normalized.json` is written before the skip-MATLAB
  bootstrap overlays the imported MATLAB settings and parity-specific cleanup
  parameters.
- Using `validated_params.json` makes the offline recomputation match the saved
  `chosen_edges.pkl` exactly.

Conclusion:

- `chosen_edges.pkl` is trustworthy for the fresh run.
- The current blocker is still candidate generation / local frontier behavior,
  not a mismatch between the chooser code path and the saved chosen-edge
  artifact.

## Candidate Coverage Snapshot

From `comparisons/20260408_current_checkout_fresh/summary.txt`:

- Candidate endpoint pairs:
  - candidate total: `2540`
  - matched MATLAB: `973`
  - missing MATLAB: `406`
- Final endpoint pairs:
  - matched: `894`
  - MATLAB-only: `485`
  - Python-only: `531`
- Chosen candidate sources:
  - frontier: `1238`
  - watershed: `187`
  - fallback: `0`

Interpretation:

- Python still misses hundreds of MATLAB endpoint pairs before cleanup begins.
- Cleanup is not the primary blocker because missing candidate pairs can never
  be recovered downstream.
- The remaining problem surface is still upstream in edge candidate generation,
  especially frontier behavior around shared neighborhoods.

## Shared-Vertex Spot Checks

Fresh candidate and chosen-edge inspection on the new run:

### Vertex 359

- Origin `359` candidates:
  - `(359, 181)` frontier, metric `-305.0`, length `15`
  - `(359, 412)` watershed, metric `-40.8`, length `22`
- Final chosen edges touching vertex `359`:
  - `(359, 1180)` frontier
  - `(359, 1568)` frontier

Takeaway:

- The origin itself does not generate the missing MATLAB-style local partner
  set.
- The final chosen neighbors touching `359` come from nearby competing origins.

### Vertex 572

- Origin `572` candidates:
  - `(572, 488)` frontier, metric `-253.9`, length `41`
- Final chosen edges touching vertex `572`:
  - `(364, 572)` watershed

Takeaway:

- The missing MATLAB-style neighborhood around `572` is absent from the
  frontier candidate pool at the origin itself.
- Downstream selection later keeps a weaker watershed alternative instead.

### Vertex 866

- Origin `866` candidates:
  - `(866, 885)` frontier, metric `-137.1`, length `17`
- Other candidates touching `866`:
  - `(810, 866)` frontier, origin `810`
  - `(394, 866)` watershed, origin `394`
  - `(701, 866)` watershed, origin `701`
- Final chosen edges touching `866`:
  - `(866, 885)`

Takeaway:

- The candidate neighborhood is active, but the expected MATLAB partners still
  never appear as Python candidates.

### Vertex 1283

- Origin `1283` candidates:
  - `(1283, 1134)` frontier, metric `-137.1`, length `18`
  - `(1283, 768)` frontier, metric `-135.4`, length `29`
  - `(1283, 1659)` watershed, metric `-3.8`, length `20`
- Final chosen edges touching `1283`:
  - `(768, 1283)`
  - `(1134, 1283)`
  - `(1283, 1659)`

Takeaway:

- The neighborhood stays active and produces multiple candidates, but the
  MATLAB-missing local partners still do not enter the candidate pool.

## Current Working Diagnosis

The fresh run reinforces the same diagnosis as the standing parity docs:

- vertex parity is exact
- stage-isolated `network` parity is already exact when exact MATLAB `edges`
  are supplied
- the remaining blocker is upstream in frontier candidate generation and local
  partner selection
- the strongest symptom is local partner substitution around active shared
  neighborhoods, not dead regions or generic graph assembly

The current best suspects remain:

- terminal-hit ownership semantics around shared root or bifurcation voxels
- parent/child invalidation plus bifurcation-half selection in live shared
  neighborhoods
- local frontier claim ordering around the first divergence points

## Practical Next Steps

1. Keep using `comparisons/20260408_current_checkout_fresh` as the staged run
   root for imported-MATLAB parity reruns.
2. Use `validated_params.json` for any offline recomputation or chooser
   instrumentation.
3. Focus the next code change in `source/slavv/core/tracing.py`, not in
   `graph.py`.
4. Localize the first per-origin frontier divergence on the shared-vertex
   cluster rather than broadening watershed heuristics.

## Related Files

- `comparisons/20260408_current_checkout_fresh/summary.txt`
- `comparisons/20260408_current_checkout_fresh/02_Output/python_results/stages/edges/candidates.pkl`
- `comparisons/20260408_current_checkout_fresh/02_Output/python_results/stages/edges/chosen_edges.pkl`
- `comparisons/20260408_current_checkout_fresh/99_Metadata/validated_params.json`
