# ADR 0005: Edge Discovery Strategy Seam

## Status
Accepted

## Context
Edge candidate generation branches on workflow mode:

- **Maintained tracing** paints vertex occupancy and runs the standard frontier tracer.
- **MATLAB-parity frontier** runs `matlab_frontier` plus watershed supplementation when `comparison_exact_network` is enabled with exact-compatible energy provenance.

Previously, this branching lived inside `resumable.py` and duplicated logic in `EdgeManager`, with a 14-callable injection surface on `extract_edges_resumable`.

## Decision
Introduce `slavv_python/processing/stages/edges/discovery.py` as the strategy seam:

1. **`CandidateManifest`** — typed wrapper for candidate payloads (`traces`, `connections`, diagnostics, lifecycle events).
2. **`EdgeDiscovery` protocol** — `discover(context) -> CandidateManifest`.
3. **Concrete strategies** — `MaintainedTracingDiscovery`, `FrontierTracingDiscovery`.
4. **`select_edge_discovery(energy_data, params)`** — chooses the strategy from energy provenance and `comparison_exact_network`.

`EdgeManager.run_resumable()` calls `select_edge_discovery()` and then handles audit JSON, parity checkpoints, selection, bridging, and finalization. `extract_edges_resumable()` in `edges.py` is a thin delegate to `EdgeManager`.

Watershed resumable extraction remains in `resumable.py` and is reached via `EdgeManager.run_watershed_resumable()`.

## Consequences
- **Single resumable entrypoint** for tracing workflows; no callable injection at the public boundary.
- **Clear extension point** for future discovery modes without growing the orchestrator.
- **Parity isolation** — MATLAB-shaped generation stays behind `FrontierTracingDiscovery` and existing `matlab_frontier` / `global_watershed` modules.
