# Parity Decision Memo 2026-04-08

## What this file is for

This is the shortest current summary of the parity story. It is the canonical
orientation note for the April 7 to April 8 investigation set.

## Read this when

- you want the current parity blocker in one page
- you need to know what is already solved versus still open
- you want the next implementation target before opening the longer audit

## Executive Summary

- Full imported-MATLAB parity still fails on the April 8 rerun:
  MATLAB edges `1379` vs Python edges `1425`, and MATLAB strands `682` vs
  Python strands `681`.
- Vertex parity is already exact.
- Network parity is already exact when Python reruns from `network` using exact
  MATLAB `edges` with `comparison_exact_network=True`.
- The remaining parity gap is not just an upstream frontier issue anymore.
  There are two active problems: the candidate pool is still wrong upstream,
  and the Python cleanup path is modeling the wrong MATLAB cleanup surface.
- The next implementation target should be the cleanup mismatch first, because
  that removes a major confounder from the remaining edge-parity work.

## Current Status

### Solved enough to trust

- Vertices are exact on the imported-MATLAB parity surface.
- Stage-isolated `network` parity is exact when exact MATLAB `edges` are
  supplied and Python reruns from `network` in parity mode.
- The saved `chosen_edges.pkl` artifact from the fresh April 8 rerun is
  internally consistent when recomputed with `validated_params.json`, so the
  saved edge artifact is trustworthy.

### Active blocker

- The fresh imported-MATLAB rerun still lands at `+46` Python edges and `-1`
  Python strands relative to MATLAB.
- Python still misses many MATLAB endpoint pairs before cleanup begins, so the
  candidate pool remains wrong upstream.
- The Python cleanup path also injects a MATLAB helper surface that the active
  `vectorize_V200.m` cleanup chain does not use.

### Superseded conclusions

- Generic downstream graph assembly is not the main blocker anymore.
- A frontier-only diagnosis is incomplete; cleanup-surface mismatch is now a
  confirmed part of the problem.
- Reopening network-stage debugging before cleanup alignment would be low
  leverage.

## Why the diagnosis changed

Two results changed the picture:

1. The stage-isolated network probe showed that Python `network` assembly can
   already converge exactly when exact MATLAB `edges` are imported and rerun
   with `comparison_exact_network=True`.
2. The code audit showed that the active MATLAB V200 cleanup path uses
   `clean_edge_pairs`, `clean_edges_vertex_degree_excess`,
   `clean_edges_orphans`, and `clean_edges_cycles`, while Python still mixes in
   `choose_edges_V200`-style conflict painting.

That means the remaining work should be interpreted as an edge-surface problem,
not a generic whole-graph problem.

## Current decision

Treat the current parity gap as a two-part edge-surface problem:

1. upstream candidate generation still diverges from MATLAB
2. downstream cleanup is still modeled against the wrong MATLAB surface

Fix the cleanup mismatch first so the next rerun measures the real residual
frontier-generation gap instead of a mixed cleanup model.

## Recommended next actions

1. Replace the current Python cleanup path with the active MATLAB V200 cleanup
   chain:
   `clean_edge_pairs`, `clean_edges_vertex_degree_excess`,
   `clean_edges_orphans`, and `clean_edges_cycles`.
2. Rerun the imported-MATLAB parity loop after that cleanup change and measure
   the remaining edge and strand gap.
3. Treat the post-cleanup residual as the true upstream frontier/candidate
   generation problem.
4. Do not spend more time on network-stage debugging or broad heuristic tuning
   until the cleanup model matches the active MATLAB path.
