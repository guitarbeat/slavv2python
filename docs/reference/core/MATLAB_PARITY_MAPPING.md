# MATLAB Parity Mapping

[Up: Reference Docs](../README.md)

This document is the maintained source map for exact imported-MATLAB parity
work in the live Python tree.

For canonical claim boundaries and the remaining work required to fully
implement the released SLAVV method in Python, see
`MATLAB_METHOD_IMPLEMENTATION_PLAN.md`.

Status labels in this document are source-audit labels, not artifact-proof
claims. Use `docs/reference/core/EXACT_PROOF_FINDINGS.md` for proof status.

## Scope

- Canonical MATLAB source lives under `external/Vectorization-Public/source/`.
- This map focuses on the imported-MATLAB `edges` and `network` stages, which
  are the active parity gap in the live Python implementation.
- Imported-MATLAB reruns already reuse preserved MATLAB energy artifacts, so
  this document does not attempt to re-audit the native Python energy backends.
- Treat any undocumented deviation on these parity surfaces as a bug.

## Exact Parity Rule

- Python parity work must reproduce the same mathematical method and algorithm
  structure as MATLAB, not just similar output counts.
- Heuristic supplements, salvage passes, reordered claim resolution, or
  simplified local workflows are not acceptable on exact-parity paths unless
  they are explicitly documented as non-parity behavior.
- Audit the current Python implementation against this map and the MATLAB source
  before making parity fixes.

## Canonical MATLAB Files

The active MATLAB sources for the imported-MATLAB parity target are:

- `vectorize_V200.m`
- `get_edges_V300.m`
- `get_edges_by_watershed.m`
- `get_edge_metric.m`
- `choose_edges_V200.m`
- `clean_edges_vertex_degree_excess.m`
- `clean_edges_orphans.m`
- `clean_edges_cycles.m`
- `add_vertices_to_edges.m`
- `get_network_V190.m`
- `sort_network_V180.m`
- `get_strand_objects.m`

## Live Python Mapping

| MATLAB surface | Live Python surface | Status | Notes |
| --- | --- | --- | --- |
| `get_edges_V300.m` and `get_edges_by_watershed.m` | `source/slavv/core/edges.py`, `source/slavv/core/_edges/standard.py`, `source/slavv/core/_edges/resumable.py`, `source/slavv/core/_edge_candidates/generate.py`, `source/slavv/core/_edge_candidates/global_watershed.py`, `source/slavv/core/_edge_candidates/common.py` | Ported on imported-MATLAB exact route; proof pending | The exact route now uses one global shared-state watershed-style search with shared `available_locations`, `pointer_map`, `d_over_r_map`, `branch_order_map`, and `vertex_adjacency_matrix`, plus MATLAB-shaped join tracing and exact V300 tolerances. Legacy non-parity tracing helpers still exist outside the exact route. |
| `get_edge_metric.m` | `source/slavv/core/_edge_primitives/metrics.py` | Source-aligned | Both use the current MATLAB `max`-energy edge metric, with `NaN` coerced to `-1000`. |
| `choose_edges_V200.m` | `source/slavv/core/_edge_selection/payloads.py`, `source/slavv/core/_edge_selection/conflict_painting.py`, `source/slavv/core/_edges/postprocess.py` | Ported on imported-MATLAB exact route; proof pending | The exact route now mirrors MATLAB pair cleanup ordering, conflict painting, pre-clean `smooth_edges_V2 -> crop_edges_V200`, degree/orphan/cycle cleanup, final `smooth_edges_V2`, and endpoint-energy normalization. Python still records extra diagnostics, but those side-channel fields do not affect the math. |
| `clean_edges_vertex_degree_excess.m` | `source/slavv/core/_edge_selection/cleanup.py` | Source-aligned; proof pending | The exact route now removes worst incident edges using MATLAB-shaped sparse edge-id lookup and descending removal order. |
| `clean_edges_orphans.m` | `source/slavv/core/_edge_selection/cleanup.py` | Source-aligned; proof pending | The exact route now mirrors MATLAB's iterative terminal-contact test against vertex terminals plus painted edge interiors, with renumbering between orphan-removal passes. |
| `clean_edges_cycles.m` | `source/slavv/core/_edge_selection/cleanup.py` | Source-aligned; proof pending | The exact route now iteratively finds cyclical connected components and removes the worst edge from each one, matching MATLAB's component-wise cleanup rather than a spanning-forest shortcut. |
| `add_vertices_to_edges.m` | `source/slavv/core/_edges/bridge_vertices.py`, `source/slavv/core/_edges/standard.py`, `source/slavv/core/_edges/resumable.py`, `source/slavv/core/graph.py` | Ported on imported-MATLAB exact route; proof pending | The exact route now detects endpoint/interior overlaps, temporarily removes unrelated edges from the bridge-reconfiguration working set, inserts bridge vertices, relabels child edges, splits parent edges, emits separate `bridge_edges`, augments network vertex counts with the inserted bridge vertices, resolves bridge targets only through a MATLAB-shaped local best-first search over edge voxels modeled on `get_edges_for_vertex(..., 'add_vertex_to_edge')`, and mirrors MATLAB `get_edge_vectors.m` for the auxiliary bridge-edge payload by exporting MATLAB-shaped `edges2vertices` plus mean bridge-edge energies. |
| `get_network_V190.m`, `sort_network_V180.m`, `get_strand_objects.m` | `source/slavv/core/graph.py` | Ported on imported-MATLAB exact route; proof pending | The exact route now isolates the degree-2 interior subgraph, forms MATLAB strand components, sorts edge order and direction, assembles ordered strand objects from edge traces, preserves full `[y, x, z, scale]` strand subscripts, then computes smoothed strands, vessel directions, and strand metrics with MATLAB-shaped helpers. |

## Current Confirmed Remaining Deviations

### 1. Exact-Parity Math Is Still Gated Behind A Special Path

The live Python tree only routes into the restored frontier parity path when
`comparison_exact_network` is enabled and the energy surface is marked as
`matlab_batch_hdf5` in `source/slavv/core/_edge_candidates/common.py`.

That means the general live Python workflow still supports non-parity code
paths, even though the imported-MATLAB exact route now follows the MATLAB
mathematical method much more closely.

### 2. Legacy Non-Parity Helpers Still Exist Outside The Exact Route

Legacy local tracing and supplement helpers still exist under:

- `source/slavv/core/_edge_candidates/frontier_trace.py`
- `source/slavv/core/_edge_candidates/frontier_resolution.py`
- `source/slavv/core/_edge_candidates/watershed_contacts.py`
- `source/slavv/core/_edge_candidates/watershed_joins.py`
- `source/slavv/core/_edge_candidates/geodesic_salvage.py`

They are acceptable as non-parity support surfaces, but they should not be
treated as the canonical imported-MATLAB method.

### 3. Source-Level Audit Still Needs Continued Verification

The exact route has been ported function-by-function, but continued line-by-line
audits against the active MATLAB sources are still required whenever the exact
route changes.

## Porting Priority

For exact imported-MATLAB parity, the highest-value remaining fixes are:

1. Keep auditing the exact route against active MATLAB sources whenever the
   imported-MATLAB path changes.
2. Verify the completed exact route against reusable imported-MATLAB staged
   runs after math-bearing changes land.

## Three-Pass Port Plan

The exact-parity port is now best tracked as three deliberate passes:

### Pass 1: Edge Discovery

- Status: Source port complete on the imported-MATLAB exact route; artifact
  proof still pending.
- The exact route now ports MATLAB `get_edges_V300.m` and
  `get_edges_by_watershed.m` into a shared-state Python workflow.
- The exact route no longer depends on the old per-origin frontier/supplement
  approximation layers for its canonical mathematical method.

### Pass 2: Edge Cleanup And Bridge Vertices

- Status: Source port complete on the imported-MATLAB exact route; artifact
  proof still pending.
- `get_edge_metric.m`, `choose_edges_V200.m`,
  `clean_edges_vertex_degree_excess.m`, `clean_edges_orphans.m`, and
  `clean_edges_cycles.m` are now ported on the exact route.
- `add_vertices_to_edges.m` now has a maintained Python equivalent for overlap
  detection, bridge-vertex insertion, parent splitting, and `bridge_edges`
  output.
- The working bridge pass now matches MATLAB's temporary restriction to the
  active parent/child edge subset before the direct local search runs.
- Bridge-target resolution now relies only on the direct local-search port
  modeled on MATLAB `get_edges_for_vertex(..., 'add_vertex_to_edge')`.

### Pass 3: Network And Strand Assembly

- Status: Source port complete on the imported-MATLAB exact route; downstream
  artifact proof still pending.
- `source/slavv/core/graph.py` now carries direct ports of
  `get_network_V190.m`, `sort_network_V180.m`, and `get_strand_objects.m`
  for the exact route.
- MATLAB strand ordering, edge direction flags, bifurcation detection, loop
  handling, strand object assembly, strand smoothing, vessel directions, and
  strand metrics are now represented on that route.
