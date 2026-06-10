# Technical Architecture

[Up: Reference Docs](../README.md)

---

## Core Engine Components

The Python implementation is built around a centralized, resumable processing engine in `slavv_python.engine`.

### The Orchestrator (`SlavvPipeline`)
The `SlavvPipeline` class is the primary entry point for running the extraction workflow. It manages the transition between stages and handles the delegation to specialized operation modules.

- **Stateless Operation**: The pipeline itself does not hold large data in memory; it retrieves and persists stage results via the `RunContext`.
- **Stage Encapsulation**: Each major phase (Energy, Vertices, Edges, Network) is encapsulated in its own operation module.
- **Typed completion**: `run()` returns `PipelineResult` (mapping-compatible for legacy `results["key"]` access).
- **StageExecutor**: Centralizes checkpoint load/save, progress, and schema wrapping for resumable stages.

### Run ledger (`RunContext` + `StageController`)
Run lifecycle lives in `slavv_python.engine.state`:

- **`run_ledger.py`** — `RunContext` (fingerprints, resume policy, snapshot persistence).
- **`stage_handle.py`** — `StageController` (checkpoint paths, `begin` / `update` / `complete`, `resume_state.json`).
- **`snapshot_lifecycle.py`** — snapshot mutation helpers (`begin_stage_snapshot`, `complete_stage_snapshot`, etc.).
- Import `RunContext` / `StageController` from `slavv_python.engine` or `slavv_python.engine.state` (not a separate barrel module).

- **Resumable State**: Fingerprints of input images and parameters determine whether a cached stage result can be reused.
- **StageController**: Each stage receives a controller that provides paths for artifacts (checkpoints, metrics, logs) and manages progress reporting.

### Typed Result Objects (`slavv_python.schema.results`)
All data passed between stages is wrapped in validated, bit-accurate dataclass models:

- `EnergyResult`: Multiscale energy volumes and metadata.
- `VertexSet`: Coordinate sets, radii, and discovery scales.
- `EdgeSet`: Traces (list of coord arrays), connectivity matrices, and endpoint energies.
- `NetworkResult`: Final topology, strands, and bifurcations.
- `PipelineResult`: Run envelope combining stage payloads and `parameters`.

### Energy stage facade (`EnergyManager`)
`slavv_python.pipeline.energy.manager` owns the Energy Field lifecycle:

- **Unified API**: Provides `run()` (ephemeral) and `run_resumable()` (persisted) entry points, abstracting multi-scale dispatch and chunking.
- **Incremental scale selection**: To prevent OOM on large volumes, the exact-route engine computes best energy and scale indices per voxel incrementally, avoiding large 4D buffers.
- **`EnergyManager.run()`** — ephemeral chunked or direct multi-scale Hessian energy → `EnergyResult`.
- **`EnergyManager.run_resumable()`** — same computation with `best_energy`, `best_scale`, and optional `energy_4d` artifacts (zarr/npy per ADR 0001).
- **`energy.py` / `resumable.py`** — thin delegates preserving public `calculate_energy_field*` imports.

### Vertex stage facade (`VertexManager`)
`slavv_python.pipeline.vertices.manager` owns the Vertex Set lifecycle:

- **Unified API**: Standardized `run()` and `run_resumable()` methods for in-memory or disk-backed extraction.
- **`VertexManager.run()`** — ephemeral scan → crop/sort → choose/paint → `VertexSet`.
- **`VertexManager.run_resumable()`** — same pipeline with `candidates.pkl`, `cropped_candidates.pkl`, `chosen_mask.pkl` artifacts.
- **`vertices/detection.py`** — MATLAB-style candidate scan and selection (no longer under `edges/`).
- **`extraction.py` / `resumable.py`** — thin delegates preserving public imports.

### Edge stage facade (`EdgeManager` + `discovery`)
The edges package exposes a deep module boundary:

- **Unified API**: Exposes `run()`, `run_resumable()`, and `discover_candidates()` to support full pipeline and audit workflows.
- **`EdgeManager.run()`** — ephemeral tracing (shared `_run_tracing()` core with resumable path).
- **`EdgeManager.run_resumable()`** — resumable tracing workflow (audit artifacts, parity checkpoints, selection, bridging, finalize).
- **`discovery.select_edge_discovery()`** — strategy seam (`MaintainedTracingDiscovery` vs `FrontierTracingDiscovery`).
- **`resumable.py`** — watershed-only per-label unit persistence.

See [ADR 0003](../../adr/0003-edge-lifecycle-manager.md) and [ADR 0005](../../adr/0005-edge-discovery-strategy-seam.md).

### Network stage facade (`NetworkManager`)
`slavv_python.pipeline.network.manager` mirrors the edge pattern:

- **Unified API**: standard `run()` and `run_resumable()` methods for graph assembly and pruning.
- **`NetworkManager.run()`** — ephemeral adjacency → prune → strand trace → `NetworkResult`.
- **`NetworkManager.run_resumable()`** — same pipeline with `adjacency.pkl`, `hair_pruned.pkl`, `cycle_pruned.pkl`, `strands.pkl` artifacts.
- **`construction.py`** — thin delegates preserving public `construct_network*` imports.

See [ADR 0006](../../adr/0006-network-lifecycle-manager.md).

### Exact parity coordinator (`ExactProofCoordinator`)
`slavv_python.analytics.parity.coordinator` centralizes exact-route workflows:

- **`prove()`** — compare normalized Python checkpoints to MATLAB oracle vectors.
- **`capture_candidates()`** — edge candidate generation via `EdgeManager.discover_candidates()` (discovery strategy seam).
- **`counts.py`** — canonical `RunCounts` extraction from reports and run directories (typed checkpoint loaders).

Parity execution helpers are split: `params_audit.py` (exact param audit/persistence), `surfaces.py` (dataset/oracle/run authority), `bootstrap.py` (init-exact-run), with `execution.py` as a compatibility facade. `reports.py` re-exports count helpers; `proofs.py` delegates to the coordinator.

### Application run envelope (`AppRunState`)
The Streamlit / shared-state layer stores **`AppRunState`** (`schema/app_run.py`) in session: a typed wrapper around `PipelineResult`, parameters, and run metadata. Dict serialization is deferred to export and share helpers only.

---

## Processing Workflow

The pipeline follows a strict linear execution order to maintain MATLAB compatibility:

1.  **Preprocessing**: Normalization and bandpass filtering of the input TIFF.
2.  **Energy Field**: Multiscale Hessian enhancement (using local tiling for large volumes).
3.  **Vertex Extraction**: Seed point discovery via local maxima suppression.
4.  **Edge Extraction**: Candidate discovery (maintained tracing or MATLAB-parity frontier), conflict-aware selection, optional bridge insertion, then finalize.
5.  **Network Construction**: Graph assembly, cycle breaking, and strand smoothing.

---

## Design Principles

### Parity-First Architecture
The engine is designed to allow "surgical" parity alignments. Logic that must match MATLAB exactly is isolated in `matlab_algorithms/` subdirectories, while the maintained Python workflow reuses these primitives.

### Memory Safety
For large 2-photon volumes (e.g. 512x512x64), the engine uses:
- **Tiled Processing**: Energy computation and vertex discovery are performed in overlapping chunks.
- **Incremental Aggregation**: Exact energy computation avoids large 4D buffers by updating best energy/scale per voxel during the multi-scale loop.
- **Kernel Pre-computation**: Scale-independent derivative kernels are computed once per chunk to reduce allocation overhead.
- **Disk-Backed Storage**: Large 4D energy stacks can be stored in `Zarr` format to avoid OOM (Out-of-Memory) errors.

### Bit-Accurate Precision
Following the **May 2026 breakthrough**, all core watershed and energy calculations are forced to `float64` to prevent tie-breaking divergences caused by lower-precision accumulation. Exact-route meshes preserve MATLAB `linspace` roundoff drift to ensure bit-perfect interpolation at octave boundaries.
