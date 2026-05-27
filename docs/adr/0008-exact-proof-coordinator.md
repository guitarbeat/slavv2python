# ADR 0008: Exact Proof Coordinator

## Status
Accepted

## Context
Exact-route parity orchestration was split across `execution.py` (~800 lines), `proofs.py` (candidate capture calling frontier internals directly), and duplicate `RunCounts` helpers in `execution.py` vs `reports.py` with incompatible report key shapes.

## Decision
1. **`ExactProofCoordinator`** (`analytics/parity/coordinator.py`) — `prove()`, `capture_candidates()`, `prepare_dest_run()`, typed loaders for exact energy/vertices.
2. **`counts.py`** — canonical `extract_matlab_counts`, `extract_source_python_counts`, `read_python_counts_from_run` (snapshot + typed checkpoint loaders).
3. **Candidate capture** — `EdgeManager.discover_candidates()` via the discovery strategy seam, not direct `_generate_edge_candidates_matlab_frontier` calls.
4. **`proofs.py`** — thin delegates; `reports.py` re-exports count helpers from `counts.py`.

## Consequences
- One locality for prove/capture orchestration and count normalization.
- Parity reruns use the same Edge Discovery interface as production.
- `run_exact_preflight` remains a stub until a later ADR defines memory-gate behavior.
