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
| `choose_edges_V200.m` | `source/core/matlab_compat/stages.py`, `source/core/edge_selection.py`, `source/core/_edge_selection/payloads.py`, `source/core/_edge_selection/conflict_painting.py`, `source/core/_edges/postprocess.py` | Ported on native-first exact route; proof pending | The chooser now sits on the canonical native exact route rather than behind a hidden native-energy override. |
| `clean_edges_vertex_degree_excess.m`, `clean_edges_orphans.m`, `clean_edges_cycles.m` | `source/core/_edge_selection/cleanup.py` | Source-aligned; proof pending | Cleanup math is ported but still downstream of unresolved edge parity. |
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

## Porting Priority

For native-first exact parity, the highest-value remaining fixes are:

1. Keep auditing the exact route against active MATLAB sources whenever the
   native exact path changes.
2. Close `edges.connections` on the native-first route before spending time on
   downstream network polish.
3. Keep `source/core/matlab_compat/` aligned with the released MATLAB stage and
   function boundaries so proof docs have a stable audit surface.
