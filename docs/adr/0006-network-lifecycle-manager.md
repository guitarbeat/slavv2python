# ADR 0006: Network Lifecycle Manager

## Status
Accepted

## Context
The network stage interleaved graph construction, hair pruning, cycle removal, strand topology, and smoothing across `construction.py` and `operations.py`. Resumable checkpoint I/O lived in a long procedural function parallel to the ephemeral `construct_network()` path.

## Decision
Introduce `NetworkManager` in `slavv_python/pipeline/network/manager.py`:

1. **`NetworkManager.run()`** — ephemeral graph build and final `NetworkResult`.
2. **`NetworkManager.run_resumable()`** — same pipeline with adjacency / hair / cycle / strand artifacts.
3. **`construct_network` / `construct_network_resumable`** — thin delegates preserving public imports.

`_network_payload()` remains in `construction.py` as shared final assembly.

## Consequences
- Symmetry with `EdgeManager` (ADR 0003).
- One locality for network-stage resume and proof work.
- `SlavvPipeline` and `StageExecutor` call `NetworkManager` directly for resumable runs.
