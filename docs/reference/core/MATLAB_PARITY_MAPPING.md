# MATLAB Parity Mapping

[Up: Reference Docs](../README.md)

This document is the maintained source map for the native-first exact route in
the live Python tree.

Use this file for MATLAB-to-Python mapping and confirmed structural deviations.
Use `MATLAB_METHOD_IMPLEMENTATION_PLAN.md` for claim boundaries and
`EXACT_PROOF_FINDINGS.md` for live proof status.

## Scope

- Canonical MATLAB source lives under `external/Vectorization-Public/source/`.
- The parity-facing Python shim now lives under `source/core/matlab_compat/`.
- The canonical exact route is `comparison_exact_network=True` with
  exact-compatible energy provenance; `python_native_hessian` is canonical and
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
| `vectorize_V200.m` | `source/core/matlab_compat/vectorize_v200.py`, `source/core/pipeline.py` | Source-aligned orchestration surface | The compat layer mirrors MATLAB stage order while delegating into the maintained modular pipeline. |
| `get_energy_V202.m` and `energy_filter_V200.m` | `source/core/matlab_compat/stages.py`, `source/core/energy.py`, `source/core/_energy/native_hessian.py`, `source/core/_energy/provenance.py` | Native exact-compatible source surface | Native matched filtering is the canonical exact-route energy implementation. |
| `get_vertices_V200.m` | `source/core/matlab_compat/stages.py`, `source/core/vertices.py`, `source/core/_vertices/extraction.py` | Source-aligned | The maintained exact route now starts from native energy rather than imported MATLAB energy. |
| `get_edges_V300.m` and `get_edges_by_watershed.m` | `source/core/matlab_compat/stages.py`, `source/core/edges.py`, `source/core/_edges/standard.py`, `source/core/_edges/resumable.py`, `source/core/_edge_candidates/generate.py`, `source/core/_edge_candidates/global_watershed.py`, `source/core/_edge_candidates/common.py` | Source-aligned with known control-flow deviations | The exact route uses MATLAB-shaped tracing and shared-state maps, but a few remaining control-flow surfaces are still documented below. |
| `get_edge_metric.m` | `source/core/matlab_compat/stages.py`, `source/analysis/_geometry/trace_ops.py`, `source/core/graph.py` | Source-aligned | The compat layer exposes a MATLAB-named wrapper while the maintained trace helpers stay modular. |
| `choose_edges_V200.m` | `source/core/matlab_compat/stages.py`, `source/core/edge_selection.py`, `source/core/_edge_selection/payloads.py`, `source/core/_edge_selection/conflict_painting.py`, `source/core/_edges/postprocess.py` | Ported; proof pending | Pre-paint filtering and chooser structure are aligned, but one remaining trace-order deviation is still tracked below. |
| `clean_edges_vertex_degree_excess.m`, `clean_edges_orphans.m`, `clean_edges_cycles.m` | `source/core/_edge_selection/cleanup.py` | Source-aligned | The current cleanup family matches the active MATLAB removal rules closely enough that it is no longer the first suspected mismatch surface. |
| `add_vertices_to_edges.m` | `source/core/matlab_compat/stages.py`, `source/core/_edges/bridge_vertices.py`, `source/core/_edges/standard.py`, `source/core/_edges/resumable.py` | Ported; proof pending | Bridge insertion remains downstream of the unresolved edge mismatch. |
| `get_network_V190.m`, `sort_network_V180.m`, `get_strand_objects.m` | `source/core/matlab_compat/stages.py`, `source/core/graph.py` | Ported; proof pending | Network decomposition and strand assembly remain downstream of edge proof. |

## Confirmed Structural Deviations Still Worth Tracking

### 1. `get_edges_V300.m` and `get_edges_by_watershed.m`: control-flow surfaces remain the main open risk

The live Python watershed path has absorbed the major pointer-lifecycle and
trace-sampling fixes, but the remaining open risk is still in control flow
rather than scalar math. The strongest surfaces are:

- frontier ordering and insertion semantics
- join reset and available-location cleanup semantics
- vertex `-Inf` sentinel lifecycle behavior
- diagnostic-era guards still affecting the canonical exact path

Treat these as the main native-first candidate-generation audit surfaces.

### 2. `choose_edges_V200.m`: randomized trace order in conflict painting

**MATLAB** uses `randperm` over the per-edge trace positions before painting.

**Python** still iterates traces sequentially in
`source/core/_edge_selection/conflict_painting.py`.

Why it matters:

- the first conflicting point encountered can change
- that can change the accept or reject order once the painted state diverges
- it is a real structural difference on an exact-parity route

Recommended direction:

- replace sequential trace iteration with a MATLAB-matching permuted index
  order, or
- explicitly demote this surface from exact-parity claims if deterministic
  replay is preferred over literal `randperm` behavior

## Aligned Surfaces That Should Not Be Reopened First

These are no longer the best first suspects when edge parity remains red:

- native energy route selection and exact-route gating
- the exact-route negative-energy cleanup gate
- antiparallel pair removal intent in chooser pre-filtering
- degree, orphan, and cycle cleanup ordering
- the reviewed size, distance, and direction penalty formulas

## Porting Priority

For native-first exact parity, the highest-value remaining work is:

1. close `edges.connections` on the native-first route before spending time on
   downstream network polish
2. keep `source/core/matlab_compat/` aligned with the released MATLAB stage and
   function boundaries so proof docs have a stable audit surface
3. use `EXACT_PROOF_FINDINGS.md` for live status and keep this file focused on
   structural mapping and deviations
