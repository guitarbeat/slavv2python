# ADR 0002: Centralized Stage Executor

## Status
Accepted

## Context
The pipeline execution logic was previously fragmented across several modules in `slavv_python/workflows/pipeline/`:
- `execution.py` managed the sequential loop.
- `resolution.py` handled the choice between resumable and fallback paths.
- `artifacts.py` managed low-level checkpoint checks.

This created a "sandwich" architecture where the core `SlavvPipeline` had to coordinate complex `Callable` chains, making it difficult to understand the end-to-end lifecycle of a stage. Modification of common behaviors (like logging or error reporting) required coordinated changes across multiple files.

## Decision
We have introduced a centralized `StageExecutor` in the `Engine` to consolidate all stage-level execution concerns.

1. **Lifecycle Management**: The executor is now the authoritative manager for checking checkpoints, running computation, persisting results, and handling progress/errors.
2. **Simplified Orchestration**: The `SlavvPipeline` interface is significantly simplified; it now delegates high-level stage definitions to the executor without needing to know the implementation details of "resumability."
3. **Unified Progress/Logging**: Execution-level metadata (timing, success/failure metrics) is now handled in one place, ensuring consistency across all stages.

## Consequences
- **Improved Locality**: All execution-related logic is concentrated in a single class, making it easier to debug or extend the engine's capabilities.
- **Reduced Fragmentation**: We can eventually decommission the shallow helper modules in `workflows/pipeline/`, reducing the cognitive load for new developers.
- **Inversion of Control**: The orchestrator now "submits" work to the executor, which is a more robust pattern for supporting future features like parallel stage execution or remote workers.
