# MATLAB Parity Mapping

[Up: Reference Docs](../README.md)

This document is the maintained source map for exact imported-MATLAB parity
work in the live Python tree.

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
| `get_edges_V300.m` and `get_edges_by_watershed.m` | `source/slavv/core/edges.py`, `source/slavv/core/_edges/standard.py`, `source/slavv/core/_edges/resumable.py`, `source/slavv/core/_edge_candidates/generate.py`, `source/slavv/core/_edge_candidates/frontier_trace.py`, `source/slavv/core/_edge_candidates/frontier_resolution.py`, `source/slavv/core/_edge_candidates/common.py` | Deviates | Python still traces one origin at a time and then merges results. MATLAB runs one global shared-state watershed-style search over the whole image with shared `available_locations`, `pointer_map`, `d_over_r_map`, `branch_order_map`, and `vertex_adjacency_matrix`. |
| `get_edge_metric.m` | `source/slavv/core/_edge_primitives/metrics.py` | Exact | Both use the current MATLAB `max`-energy edge metric, with `NaN` coerced to `-1000`. |
| `choose_edges_V200.m` | `source/slavv/core/_edge_selection/payloads.py`, `source/slavv/core/_edge_selection/conflict_painting.py` | Close with deviations | The self-edge, dangling-edge, negative-energy, directed-pair, antiparallel-pair, and conflict-painting steps are MATLAB-shaped. Python still adds a length-based tie-break before metric sorting and carries extra source-aware diagnostics that do not exist in MATLAB. |
| `clean_edges_vertex_degree_excess.m` | `source/slavv/core/_edge_selection/cleanup.py` | Close | Python removes the worst incident edges in best-to-worst order, which is structurally close to MATLAB's sparse edge-id removal. |
| `clean_edges_orphans.m` | `source/slavv/core/_edge_selection/cleanup.py` | Close | Python mirrors MATLAB's iterative removal of edges whose terminal voxels touch neither a vertex nor any interior edge voxel. |
| `clean_edges_cycles.m` | `source/slavv/core/_edge_selection/cleanup.py` | Deviates | MATLAB iteratively removes the worst edge from each cyclical connected component. Python currently uses a simple spanning-forest union-find pass that drops the first cycle-closing edge it encounters. |
| `add_vertices_to_edges.m` | No live Python equivalent | Missing | The live Python tree does not currently insert bridge vertices where child edges meet parent edges. |
| `get_network_V190.m`, `sort_network_V180.m`, `get_strand_objects.m` | `source/slavv/core/graph.py` | Deviates | Python constructs a generic sparse graph, walks unvisited neighbors in sorted vertex order, and returns vertex-index strands. MATLAB builds strand interiors from the degree-2 subgraph, sorts edges within each strand, tracks edge direction, and assembles ordered strand objects from edge traces. |

## Current Confirmed Deviations

### 1. Edge Discovery Is Still Local, Not Global

MATLAB `get_edges_by_watershed.m` performs one global best-first watershed-like
claim process across all vertices at once. The live Python parity path in
`frontier_trace.py` still builds separate local pointer and availability maps
for each origin vertex, then combines those results afterward.

This is the largest known algorithmic deviation in the `edges` stage.

### 2. Exact-Parity Edge Discovery Is Still Gated Behind A Special Path

The live Python tree only routes into the restored frontier parity path when
`comparison_exact_network` is enabled and the energy surface is marked as
`matlab_batch_hdf5` in `source/slavv/core/_edge_candidates/common.py`.

That means the general live Python edge workflow is still a different algorithm
surface from the MATLAB parity path.

### 3. Python Still Uses Post-Hoc Supplement And Salvage Layers

After frontier tracing, the live Python parity path can still add candidates
through:

- `source/slavv/core/_edge_candidates/watershed_contacts.py`
- `source/slavv/core/_edge_candidates/watershed_joins.py`
- `source/slavv/core/_edge_candidates/geodesic_salvage.py`

These are approximation layers around the restored frontier path. They are not
the same as MATLAB's core `get_edges_by_watershed.m` discovery loop.

### 4. Candidate Pair Filtering Is Not Yet A Pure MATLAB Port

`source/slavv/core/_edge_selection/payloads.py` performs a stable length sort
before the stable metric sort. MATLAB `choose_edges_V200.m` sorts by edge
metric only, then relies on stable `unique` and `intersect` behavior.

This is smaller than the discovery gap, but it is still a real deviation.

### 5. Cycle Cleanup Is Not MATLAB-Equivalent

`source/slavv/core/_edge_selection/cleanup.py` removes cycle-closing edges
using a single spanning-forest pass. MATLAB `clean_edges_cycles.m` repeatedly
finds cyclical connected components and removes the worst edge from each one.

This can change both final edge membership and downstream strand topology.

### 6. Bridge-Vertex Insertion Is Missing

MATLAB `add_vertices_to_edges.m` can insert new vertices where child edges meet
their parents and then split affected parent edges. The live Python tree has no
maintained equivalent pass.

This is a likely contributor to downstream strand mismatch even when edge counts
improve.

### 7. Network And Strand Assembly Are Not MATLAB-Equivalent

The live Python network stage in `source/slavv/core/graph.py` is a generic
sparse-graph walk. MATLAB instead:

- isolates the degree-2 interior subgraph in `get_network_V190.m`
- forms strand connected components from that interior-only graph
- sorts strand edges and direction in `sort_network_V180.m`
- assembles ordered strand voxel traces in `get_strand_objects.m`

The current Python network stage does not reproduce that method 1:1.

## Porting Priority

For exact imported-MATLAB parity, the highest-value fixes are:

1. Replace the per-origin frontier tracer with a direct Python port of the
   global shared-state `get_edges_by_watershed.m` method.
2. Remove watershed/geodesic supplement and salvage layers from the exact-parity
   path once the direct port exists.
3. Port `add_vertices_to_edges.m` semantics into the exact-parity path.
4. Replace the current network/strand construction path with a direct port of
   `get_network_V190.m`, `sort_network_V180.m`, and `get_strand_objects.m`.
5. Remove remaining Python-only tie-breakers from candidate pair filtering.

## Three-Pass Port Plan

The exact-parity port should proceed in three deliberate passes:

### Pass 1: Edge Discovery

- Port MATLAB `get_edges_V300.m` and `get_edges_by_watershed.m` into a single
  shared-state Python workflow.
- Replace the per-origin frontier trace path on the exact-parity route.
- Remove post-hoc watershed and geodesic supplement/salvage behavior from the
  exact-parity route once the direct port is in place.
- Preserve MATLAB ordering, shared pointer ownership, adjacency gating, branch
  bookkeeping, and edge-budget semantics exactly.

### Pass 2: Edge Cleanup And Bridge Vertices

- Match `get_edge_metric.m` and `choose_edges_V200.m` behavior exactly in the
  exact-parity path.
- Port `clean_edges_vertex_degree_excess.m`,
  `clean_edges_orphans.m`, and `clean_edges_cycles.m` exactly.
- Port `add_vertices_to_edges.m` so parent/child meeting points create bridge
  vertices and split parent edges when MATLAB does.

### Pass 3: Network And Strand Assembly

- Replace the current generic sparse-graph walk in `source/slavv/core/graph.py`
  for the exact-parity route with direct ports of `get_network_V190.m`,
  `sort_network_V180.m`, and `get_strand_objects.m`.
- Match MATLAB strand ordering, edge direction flags, bifurcation detection,
  loop handling, and strand object assembly exactly.
- Verify the completed route against reusable imported-MATLAB staged runs using
  `dev/scripts/cli/parity_experiment.py`.
