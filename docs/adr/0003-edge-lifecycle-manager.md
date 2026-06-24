# ADR 0003: Edge Lifecycle Manager

## Status
Accepted (implemented 2026-05-27)

## Context
The "Edges" stage is the most algorithmically complex part of the SLAVV pipeline, involving multiple discovery strategies (Tracing vs. Watershed), conflict-aware selection (Conflict Painting), and structural post-processing (Bridge Insertion). 

Prior to this decision, the orchestration of these steps was leaked into the core `SlavvPipeline` and its resumable wrappers. This resulted in:
1. **Shallow Interfaces**: The `extract_edges_resumable` function required over 14 keyword arguments, most of which were internal helpers from the same package.
2. **Leaked Implementation Details**: The top-level orchestrator had to manage the state required for "bridging" and "painting," which are internal to the edge extraction's goal.
3. **Fragile Coupling**: Changes to the internal lifecycle of edge discovery often required coordinated changes in the orchestration layer.

## Decision
We have introduced a consolidated `EdgeManager` in `slavv_python/pipeline/edges/manager.py` to act as a deep facade for the entire edge lifecycle.

1. **Encapsulated Lifecycle**: The manager handles the transition from candidate generation to selection and final bridging internally.
2. **High-Level Contract**: The manager accepts and returns the "Deep" schema objects (`EnergyResult`, `VertexSet`, `EdgeSet`), abstracting away the low-level dictionary structures used by internal algorithms.
3. **Internalized Logic**: Preparatory steps, such as painting vertex occupancy images, are now internal implementation details of the manager.

## Implementation (2026-05-27)
- `EdgeManager.run()` and `EdgeManager.run_resumable()` share `_run_tracing()`; resumable-only audit/checkpoint steps run when a real `StageController` is provided.
- `EdgeManager.run_resumable()` owns the full resumable lifecycle: candidate audit JSON, parity candidate checkpoints, frontier lifecycle artifacts, selection, bridging, and finalization.
- Removed `extraction_standard.py`; ephemeral orchestration calls `EdgeManager.run()` directly.
- `extract_edges_resumable()` delegates to `EdgeManager`; the former 14-callable `resumable.extract_edges_resumable` surface was removed.
- Candidate generation is routed through the discovery strategy seam ([ADR 0005](0005-edge-discovery-strategy-seam.md)).
- `resumable.py` retains only watershed per-label unit persistence.

## Consequences
- **Reduced Surface Area**: The public interface for edge extraction is now a single, high-leverage method call.
- **Improved Maintainability**: Algorithmic changes to the edge discovery process (e.g., swapping a selection heuristic) can now be made entirely within the `edges` package without affecting the orchestrator.
- **Enhanced Testability**: The `EdgeManager` provides a clear point for integration testing of the entire edge stage in isolation from the rest of the pipeline.
- **Codebase Navigability**: The "center of gravity" for the most complex stage of the project is now explicitly defined and documented.
