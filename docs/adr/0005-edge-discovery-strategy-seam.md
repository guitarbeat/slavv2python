# ADR 0005: Edge Discovery Strategy Seam

## Status
Accepted

## Context
Edge candidate generation branches on workflow mode:

- **Maintained tracing** paints vertex occupancy and runs the standard frontier tracer.
- **MATLAB-parity frontier** runs `matlab_get_edges_v300_frontier` plus watershed supplementation when `comparison_exact_network` is enabled with exact-compatible energy provenance.

Previously, this branching lived inside `resumable.py` and duplicated logic in `EdgeManager`, with a 14-callable injection surface on `extract_edges_resumable`.

## Decision
Introduce `slavv_python/pipeline/edges/discovery.py` as the strategy seam:

1. **`CandidateManifest`** — typed wrapper for candidate payloads (`traces`, `connections`, diagnostics, lifecycle events).
2. **`EdgeDiscovery` protocol** — `discover(context) -> CandidateManifest`.
3. **Concrete strategies** — `MaintainedTracingDiscovery`, `FrontierTracingDiscovery`.
4. **`select_edge_discovery(energy_data, params)`** — chooses the strategy from energy provenance and `comparison_exact_network`.

`EdgeManager.run_resumable()` calls `select_edge_discovery()` and then handles audit JSON, parity checkpoints, selection, bridging, and finalization. `extract_edges_resumable()` in `edges.py` is a thin delegate to `EdgeManager`.

Watershed resumable extraction remains in `resumable.py` and is reached via `EdgeManager.run_watershed_resumable()`.

## Consequences
- **Single resumable entrypoint** for tracing workflows; no callable injection at the public boundary.
- **Clear extension point** for future discovery modes without growing the orchestrator.
- **Parity isolation** — MATLAB-shaped generation stays behind Watershed Discovery and existing `matlab_get_edges_v300_frontier` / `matlab_get_edges_by_watershed` modules.

## Addendum (2026-07-10): Domain-aligned strategy names

Architecture review found a **name trap**: domain glossary uses **Tracing Discovery** / **Watershed Discovery**, while code used `MaintainedTracingDiscovery` / `FrontierTracingDiscovery`, and package exports highlighted skimage `extract_edges_watershed` as if it were the cert path.

**Canonical class names** (glossary-aligned):

| Domain term | Class | Notes |
|-------------|-------|-------|
| Tracing Discovery | `TracingDiscovery` | Paper Path; legacy alias `MaintainedTracingDiscovery` |
| Watershed Discovery | `WatershedDiscovery` | Exact Route; legacy alias `FrontierTracingDiscovery` |

**Predicate:** `_use_watershed_discovery` (legacy alias `_use_matlab_frontier_tracer`).

**Not** the discovery seam: `extract_edges_watershed` / `naive_watershed` remain skimage label-adjacency helpers and are documented as non-certification.

**Unchanged wire format:** diagnostic keys and `connection_sources` values such as `"frontier"` stay stable for checkpoints and audits.

**Rejected:** mass-renaming `matlab_*` modules or artifact field strings (high churn, no navigability win over class renames + package map).

## Addendum (2026-07-10): CandidateManifest deep module

Architecture review candidate #2: candidate payload helpers were scattered across `discovery.py`, shallow `candidate_manifest.py` (dict append only), and `candidate_payload.py` (reorder/endpoint pass-throughs that late-imported the type).

**Decision:** `candidate_manifest.py` owns the deep `CandidateManifest` interface:

- `from_payload` / `to_payload` / `apply_to_dict` / `append_unit` / `append_unit_into_dict`
- `reordered` / `endpoint_pair_set` / `frontier_origin_counts`
- `normalize_candidate_connection_sources`, `reorder_candidate_payload`, endpoint helpers

`discovery.py` keeps the strategy seam only and re-exports `CandidateManifest` for the ADR 0005 public surface. `candidate_payload.py` remains a thin compatibility re-export barrel.

**Unchanged:** checkpoint dict field names and connection_source wire strings.
