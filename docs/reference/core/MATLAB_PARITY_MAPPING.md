# MATLAB Parity Mapping

[Up: Reference Docs](../README.md)

This document is the maintained source map for the native-first exact route in
the live Python tree.

For canonical claim boundaries and the remaining work required to fully
implement the released SLAVV method in Python, see
`MATLAB_METHOD_IMPLEMENTATION_PLAN.md`.

Status labels in this document are source-audit labels, not artifact-proof
claims. Use `docs/reference/core/EXACT_PROOF_FINDINGS.md` for proof status.

## Scope

- Canonical MATLAB source lives under `external/Vectorization-Public/source/`.
- The parity-facing Python shim now lives under `source/core/matlab_compat/`.
- The canonical exact route is `comparison_exact_network=True` with exact-
  compatible energy provenance; `python_native_hessian` is canonical and
  `matlab_batch_hdf5` remains historical-compatible.
- Preserved MATLAB vectors are still the oracle artifacts for proof.
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

The active MATLAB sources for the native-first exact target are:

- `vectorize_V200.m`
- `get_energy_V202.m`
- `energy_filter_V200.m`
- `get_vertices_V200.m`
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
| `vectorize_V200.m` | `source/core/matlab_compat/vectorize_v200.py`, `source/core/pipeline.py` | Source-aligned orchestration surface | The compat layer mirrors MATLAB stage order while delegating into the maintained modular Python pipeline. |
| `get_energy_V202.m` and `energy_filter_V200.m` | `source/core/matlab_compat/stages.py`, `source/core/energy.py`, `source/core/_energy/native_hessian.py`, `source/core/_energy/provenance.py` | Native exact-compatible source surface | Native matched filtering is now the canonical exact-route energy implementation. |
| `get_vertices_V200.m` | `source/core/matlab_compat/stages.py`, `source/core/vertices.py`, `source/core/_vertices/extraction.py` | Source-aligned; proof pending on native exact route | The maintained exact route now starts from native energy rather than imported MATLAB energy. |
| `get_edges_V300.m` and `get_edges_by_watershed.m` | `source/core/matlab_compat/stages.py`, `source/core/edges.py`, `source/core/_edges/standard.py`, `source/core/_edges/resumable.py`, `source/core/_edge_candidates/generate.py`, `source/core/_edge_candidates/global_watershed.py`, `source/core/_edge_candidates/common.py` | **High-Performance Port (Source-Aligned)** | The exact route uses the MATLAB-style **heapq-accelerated O(log N) frontier** and **flat-first 1D Fortran architecture**. Scale-tolerance derivation now follows MATLAB's first-two-radii formula, and trace-back sampling uses direct linear offsets. |
| `get_edge_metric.m` | `source/core/matlab_compat/stages.py`, `source/analysis/_geometry/trace_ops.py`, `source/core/graph.py` | Source-aligned | The compat layer exposes a MATLAB-named wrapper while the maintained trace metric helpers stay modular. |
| `choose_edges_V200.m` | `source/core/matlab_compat/stages.py`, `source/core/edge_selection.py`, `source/core/_edge_selection/payloads.py`, `source/core/_edge_selection/conflict_painting.py`, `source/core/_edges/postprocess.py` | Ported on native-first exact route; proof pending | Pre-paint filtering (negative-energy gate, deduplication, antiparallel removal) is aligned. **Known deviation**: the painting loop iterates the trace sequentially; MATLAB uses `randperm` per edge. See Deviation #3. |
| `clean_edges_vertex_degree_excess.m`, `clean_edges_orphans.m`, `clean_edges_cycles.m` | `source/core/_edge_selection/cleanup.py` | Source-aligned; proof pending | All three cleanup functions are structurally aligned with MATLAB after the April 2026 audit. Removal order (degree), terminal union (orphans), and worst-edge-per-component (cycles) all match. See Deviations #6–9. |
| `add_vertices_to_edges.m` | `source/core/matlab_compat/stages.py`, `source/core/_edges/bridge_vertices.py`, `source/core/_edges/standard.py`, `source/core/_edges/resumable.py` | Ported on native-first exact route; proof pending | Bridge insertion remains downstream of the unresolved edge mismatch. |
| `get_network_V190.m`, `sort_network_V180.m`, `get_strand_objects.m` | `source/core/matlab_compat/stages.py`, `source/core/graph.py` | Ported on native-first exact route; proof pending | MATLAB-shaped network decomposition and strand assembly helpers are maintained directly in `graph.py`. |

## Current Confirmed Remaining Deviations

### 1. The canonical route is native-first, but historical imported-MATLAB compatibility still exists

The live Python tree now routes into the MATLAB-style frontier path when
`comparison_exact_network` is enabled and the energy surface is exact-
compatible. The canonical provenance is `python_native_hessian`; preserved
`matlab_batch_hdf5` surfaces remain supported for historical replay and oracle
comparison.

### 2. Downstream proof is still open after the energy cutover

The major remaining gap is no longer native energy provenance. The current open
work is downstream exact proof for:

- candidate generation
- edge choosing and cleanup
- bridge insertion
- network / strand assembly

### 3. `choose_edges_V200.m` — conflict-painting loop: randomised trace order (DEVIATION)

**MATLAB** (`choose_edges_V200.m`, painting loop):
```matlab
edge_position_index_range = uint16( randperm( degrees_of_edges( edge_index )));
for edge_position_index = edge_position_index_range
```
MATLAB visits each point along the trace in a **random permutation** order on
every edge evaluation.

**Python** (`conflict_painting.py`, `_choose_edges_matlab_style`):
```python
for point_index, point in enumerate(trace):
```
Python iterates the trace in **sequential order** (index 0 → end).

**Impact**: When a trace has an interior conflict, MATLAB may detect it at a
different point than Python, which can change which edge is rejected first when
two edges share overlapping influence volumes. This is a structural deviation on
the exact-parity path. It does not affect the conflict test itself (any
conflicting point causes rejection), but it can change the *first* conflicting
point found and therefore the order in which edges are accepted or rejected when
the painted image state differs between the two orderings.

**Recommended fix**: Replace the sequential `enumerate(trace)` loop with a
randomly permuted index range using `np.random.permutation(len(trace))`, seeded
deterministically per edge to match MATLAB's per-edge `randperm` call. Note
that MATLAB's `randperm` is not seeded per-edge in the source — it uses the
global RNG state — so exact reproduction requires either accepting this as a
known non-deterministic deviation or using a fixed seed for the exact route.

### 4. `choose_edges_V200.m` — pre-paint filter: negative-energy gate is MATLAB-canonical

**MATLAB** (`choose_edges_V200.m`):
```matlab
indices_of_negative_energy_edges = find( max_edge_energies < 0 );
...
mean_edge_energies    =     mean_edge_energies( indices_of_negative_energy_edges );
original_edge_indices = original_edge_indices(  indices_of_negative_energy_edges );
edges2vertices        =   edges2vertices(       indices_of_negative_energy_edges, : );
```
MATLAB **keeps** only edges whose maximum energy along the trace is negative
(i.e. `max < 0`), discarding any edge that ever attained a non-negative energy.

**Python** (`payloads.py`, `prepare_candidate_indices_for_cleanup`):
```python
reject_nonnegative_energy_edges: bool = True,
...
nonnegative_max = np.array([
    np.nanmax(...) >= 0
    for index in filtered_indices
], dtype=bool)
filtered_indices = filtered_indices[~nonnegative_max]
```
Python's gate is equivalent when `reject_nonnegative_energy_edges=True`. On the
exact route this flag is set to `False` for `matlab_global_watershed_exact`
candidates (`conflict_painting.py` line: `reject_nonnegative_energy_edges=not
bool(candidates.get("matlab_global_watershed_exact", False))`).

**Status**: The gate is correctly disabled for the exact watershed route.
Confirmed aligned for the non-exact path. No fix needed here; document only.

### 5. `choose_edges_V200.m` — pre-paint deduplication: antiparallel pair removal

**MATLAB** (`choose_edges_V200.m`):
```matlab
[ ~, mutual_edges_sorted_indices, mutual_edges_indices_antiparallel_to_sorted ] ...
    = intersect([ edges2vertices( :, 1 ), edges2vertices( :, 2 )],  ...
                [ edges2vertices( :, 2 ), edges2vertices( :, 1 )],  ...
                'rows', 'stable'                                );
logical_of_worse_of_mutual_edge_pairs
    = mutual_edges_sorted_indices > mutual_edges_indices_antiparallel_to_sorted;
original_edge_indices( mutual_edges_sorted_indices( logical_of_worse_of_mutual_edge_pairs )) = [];
```
MATLAB removes the **higher-index** (worse-energy) edge from each antiparallel
pair (A→B vs B→A) before the painting loop.

**Python** (`payloads.py`, `prepare_candidate_indices_for_cleanup`):
```python
undirected_seen: set[tuple[int, int]] = set()
for index in directed_indices:
    pair_u = (min(start, end), max(start, end))
    if pair_u in undirected_seen:
        diagnostics["antiparallel_pair_count"] += 1
        continue
    undirected_seen.add(pair_u)
    filtered_unique_indices.append(int(index))
```
Python keeps the **first** undirected pair seen in the energy-sorted order,
which is equivalent to keeping the lower-energy direction — matching MATLAB's
intent of keeping the better of the two mutual edges.

**Status**: Aligned in intent. The Python approach is equivalent because
`directed_indices` is already sorted by energy ascending before the undirected
deduplication pass.

### 6. `clean_edges_vertex_degree_excess.m` — removal order: highest-index edges removed first

**MATLAB**:
```matlab
edges_at_vertex_descending = sort( edges_at_vertex, 'descend' );
ordinates_of_edges_to_remove = 1 : vertex_excess_degrees( vertex_index );
edges_to_remove(...) = edges_at_vertex_descending( ordinates_of_edges_to_remove );
```
MATLAB removes the edges with the **highest index values** (worst energy, since
edges are sorted best-first before cleanup) at each excess-degree vertex.

**Python** (`cleanup.py`, `clean_edges_vertex_degree_excess_python`):
```python
edges_at_vertex_descending = np.sort(edges_at_vertex)[::-1]
excess_degree = int(vertex_excess_degrees[vertex_index])
edges_to_remove.extend(edges_at_vertex_descending[:excess_degree].astype(int).tolist())
```
Python sorts descending and removes the top `excess_degree` entries — identical
logic to MATLAB.

**Status**: Aligned. No fix needed.

### 7. `clean_edges_cycles.m` — cycle pruning: worst edge per component

**MATLAB**:
```matlab
edges_to_remove(...) = max( edge_indices_in_cycles );
```
Removes the edge with the **maximum index** (worst energy) from each cycle
connected component.

**Python** (`cleanup.py`, `clean_edges_cycles_python`):
```python
worst_edge_id = int(np.max(component_lookup.data))
removed_edge_ids.append(worst_edge_id)
```
Identical: removes the maximum-index edge per cycle component.

**Status**: Aligned. No fix needed.

### 8. `clean_edges_cycles.m` — vertex pruning between iterations

**MATLAB** (after each while-loop iteration):
```matlab
vertices_to_be_truncated = 1 : number_of_truncated_vertices;
vertices_to_be_truncated( cell2mat( vertices_in_cycles )) = [];
adjacency_matrix( vertices_to_be_truncated, : ) = [];
edge_lookup_table( vertices_to_be_truncated, : ) = [];
adjacency_matrix( :, vertices_to_be_truncated ) = [];
edge_lookup_table( :, vertices_to_be_truncated ) = [];
```
After removing the worst edge from each cycle component, MATLAB **prunes all
vertices that were not part of any cycle** from the working matrices before the
next iteration. This progressively shrinks the working set to only cycle-
involved vertices.

**Python** (`cleanup.py`, `clean_edges_cycles_python`):
```python
adjacency = adjacency[cycle_vertex_mask][:, cycle_vertex_mask].tocsr()
edge_lookup = edge_lookup[cycle_vertex_mask][:, cycle_vertex_mask].tocsr()
```
Python retains only vertices that were in a cycle component (`cycle_vertex_mask`
is set for all vertices in any component with `> 1` node). This is equivalent
to MATLAB's pruning of non-cycle vertices between iterations.

**Status**: Aligned in intent. The Python mask correctly retains only cycle-
involved vertices between iterations, matching MATLAB's progressive truncation.

### 9. `clean_edges_orphans.m` — terminal check: interior union vs. repeat-location union

**MATLAB** (active path):
```matlab
[ ~, orphan_terminal_indices ] = setdiff( exterior_edge_locations( : ), ...
    union( interior_edge_locations, vertex_locations ));
```
MATLAB checks whether each terminal location appears in the union of **interior
edge locations** and **vertex locations**.

**Python** (`cleanup.py`, `clean_edges_orphans_python`):
```python
union_locations = np.union1d(
    interior_edge_locations, np.fromiter(vertex_locations, dtype=np.int64)
)
orphan_value_mask = ~np.isin(unique_exterior_locations, union_locations)
```
Python uses the same union of interior locations and vertex locations.

**Status**: Aligned with the active MATLAB path. (The MATLAB source contains a
commented-out older variant using `edge_location_repeats` instead of
`interior_edge_locations`; the active code uses `interior_edge_locations` and
Python matches it.)

## Porting Priority

For native-first exact parity, the highest-value remaining fixes are:

1. **Fix the randomised trace-order deviation** (Deviation #3 above) in
   `conflict_painting.py`. This is the only structural deviation found in the
   conflict-painting loop. Replace sequential trace iteration with a per-edge
   permuted order. Decide whether to use a fixed seed for the exact route or
   accept this as a documented non-deterministic deviation.
2. Close `edges.connections` on the native-first route before spending time on
   downstream network polish.
3. Keep `source/core/matlab_compat/` aligned with the released MATLAB stage and
   function boundaries so proof docs have a stable audit surface.
