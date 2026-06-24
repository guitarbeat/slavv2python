# ADR 0007: Vertex Lifecycle Manager

## Status
Accepted

## Context
Vertex extraction interleaved MATLAB-style scan, crop/sort, and choose/paint across `extraction.py` and `resumable.py`. Vertex detection logic lived under `edges/candidate_detection.py`, inverting the package dependency (Vertex Set stage importing from Edge Discovery).

## Decision
Introduce `VertexManager` in `slavv_python/pipeline/vertices/manager.py`:

1. **`VertexManager.run()`** — ephemeral scan → crop/sort → choose/paint → `VertexSet`.
2. **`VertexManager.run_resumable()`** — same pipeline with `candidates.pkl`, `cropped_candidates.pkl`, `chosen_mask.pkl` artifacts.
3. **`vertices/detection.py`** — MATLAB-style candidate scan and selection (moved from `edges/candidate_detection.py`).
4. **`extract_vertices` / `extract_vertices_resumable`** — thin delegates; `edges/candidate_detection.py` re-exports for compatibility.

## Consequences
- Symmetry with `EdgeManager` (ADR 0003) and `NetworkManager` (ADR 0006).
- Vertex Set stage owns its discovery implementation; edges no longer host vertex algorithms.
- `SlavvPipeline` calls `VertexManager` directly for resumable runs.
