# MATLAB Parity Mapping

[Up: Reference Docs](../README.md)

This document is the maintained slavv_python map for the native-first exact route in
the live Python tree.

Use this file for MATLAB-to-Python mapping and confirmed structural deviations.
Use `MATLAB_METHOD_IMPLEMENTATION_PLAN.md` for claim boundaries and
`EXACT_PROOF_FINDINGS.md` for live proof status.

## Scope

- Canonical MATLAB slavv_python lives under `external/Vectorization-Public/slavv_python/`.
- The parity-facing Python shim now lives under `slavv_python/core/edges/matlab_algorithms/`.
- The canonical exact route is `comparison_exact_network=True` with
  exact-compatible energy provenance; `python_native_hessian` is the only
  accepted exact-route energy provenance.
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
| `vectorize_V200.m` | `slavv_python/core/edges/matlab_algorithms/vectorize_v200.py`, `slavv_python/core/pipeline.py` | Source-aligned orchestration surface | The compat layer mirrors MATLAB stage order while delegating into the maintained modular pipeline. |
| `get_energy_V202.m` and `energy_filter_V200.m` | `slavv_python/core/edges/matlab_algorithms/stages.py`, `slavv_python/core/energy/energy.py`, `slavv_python/core/energy/hessian_response.py`, `slavv_python/core/energy/provenance.py` | Native exact-compatible slavv_python surface | Native matched filtering is the canonical exact-route energy implementation. |
| `get_vertices_V200.m` | `slavv_python/core/edges/matlab_algorithms/stages.py`, `slavv_python/core/vertices/vertices.py`, `slavv_python/core/vertices/extraction.py` | Source-aligned | The maintained exact route now starts from native energy rather than imported MATLAB energy. |
| `get_edges_V300.m` and `get_edges_by_watershed.m` | `slavv_python/core/edges/matlab_algorithms/stages.py`, `slavv_python/core/edges/edges.py`, `slavv_python/core/edges/candidate_generation.py`, `slavv_python/core/edges/matlab_frontier.py`, `slavv_python/core/edges/tracing.py`, `slavv_python/core/edges/resumable.py` | Source-aligned with known control-flow deviations | Preferred grouped facades delegate into modular implementation modules under `slavv_python/core/edges/` where the remaining exact-route work still lives. |
| `get_edge_metric.m` | `slavv_python/core/edges/matlab_algorithms/stages.py`, `slavv_python/analysis/trace_ops.py`, `slavv_python/core/network.py` | Source-aligned | The compat layer exposes a MATLAB-named wrapper while the maintained trace helpers stay modular. |
| `choose_edges_V200.m` | `slavv_python/core/edges/matlab_algorithms/stages.py`, `slavv_python/core/edges/selection.py` | Ported; proof pending | Pre-paint filtering and chooser structure are aligned, but one remaining trace-order deviation is still tracked below. |
| `clean_edges_vertex_degree_excess.m`, `clean_edges_orphans.m`, `clean_edges_cycles.m` | `slavv_python/core/edges/cleanup.py` | Source-aligned | The preferred cleanup surface delegates into the historical cleanup implementation, which is no longer the first suspected mismatch surface. |
| `add_vertices_to_edges.m` | `slavv_python/core/edges/matlab_algorithms/stages.py`, `slavv_python/core/edges/bridge_insertion.py`, `slavv_python/core/edges/extraction_standard.py`, `slavv_python/core/edges/resumable.py` | Ported; proof pending | Bridge insertion remains downstream of the unresolved edge mismatch. |
| `get_network_V190.m`, `sort_network_V180.m`, `get_strand_objects.m` | `slavv_python/core/edges/matlab_algorithms/stages.py`, `slavv_python/core/network.py`, `slavv_python/core/graph/` | Ported; proof pending | `slavv_python/core/network.py` is the preferred network surface; strand assembly remains downstream of edge proof. |

## Confirmed Structural Deviations Still Worth Tracking

## Red Herrings To Avoid

These are now explicit parity-investigation pitfalls:

- Do not treat preserved `settings/*.mat` files as the complete MATLAB method
  surface.
  Important edge-stage constants such as `step_size_per_origin_radius`,
  `max_edge_energy`, `edge_number_tolerance`, `distance_tolerance`, and
  `radius_tolerance` can live only in released MATLAB slavv_python and still need to
  be persisted into the exact params surface.
- Do not feed the vertex fields embedded inside raw `edges*.mat` into upstream
  watershed candidate generation.
  The standalone curated vertex artifact is the correct upstream vertex surface;
  the embedded edge-file vertices are downstream and can include added bridge
  vertices.
- Do not treat candidate coverage against the final preserved `edges` artifact
  as an isolated proof of upstream watershed parity.
  The final edge artifact already includes downstream bridge insertion effects.
- Do not assume the preserved 2019 oracle batch and the later public
  `Vectorization-Public` slavv_python are one-to-one in code vintage.
  When artifact behavior and later slavv_python comments disagree, record the
  discrepancy explicitly and avoid treating either side as silently
  self-evident.

### 1. `get_edges_V300.m` and `get_edges_by_watershed.m`: control-flow surfaces aligned

The live Python watershed path has absorbed the major pointer-lifecycle, trace-sampling, and frontier-ordering fixes. The following control-flow surfaces are now aligned with MATLAB:

- ✅ frontier ordering and insertion semantics (r/R normalization)
- ✅ join reset and available-location cleanup semantics
- ✅ vertex `-Inf` sentinel lifecycle behavior
- ✅ energy map integrity (stopped penalty propagation to shared map)

### 2. `choose_edges_V200.m`: randomized trace order in conflict painting - ✅ FIXED

**Status**: Fixed (2026-05-05)

**MATLAB** uses `randperm` over the per-edge trace positions before painting.

**Python** now uses a seeded `np.random.default_rng(seed).permutation()` in `slavv_python/core/edges/selection.py` to match this behavior while maintaining determinism for parity testing.

Why it matters:
- ensures literal chooser parity by matching the conflict detection sequence
- prevents accept/reject divergence caused by trace iteration order


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
2. keep `slavv_python/core/edges/matlab_algorithms/` aligned with the released MATLAB stage and
   function boundaries so proof docs have a stable audit surface
3. use `EXACT_PROOF_FINDINGS.md` for live status and keep this file focused on
   structural mapping and deviations
