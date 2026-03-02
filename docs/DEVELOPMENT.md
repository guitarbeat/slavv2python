# Development Guide

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core Pipeline (`pipeline.py`) | Complete | Energy -> Vertices -> Edges -> Network |
| I/O (`io_utils.py`) | Complete | MAT, CASX, VMV, CSV, JSON, DICOM, TIFF |
| Visualization (`visualization.py`) | Complete | 2D/3D Plotly, animations |
| ML Curation (`ml_curator.py`) | Functional | Logistic/RF classifiers |
| Test Suite | Active | Unit + integration coverage |

## Repository Organization

### Top-Level Layout

- `source/slavv/`: production package code only.
- `tests/`: all automated tests.
- `docs/`: developer and user documentation.
- `workspace/`: local workflows, notebooks, scripts, and generated local temp artifacts.
- `external/`: upstream or third-party external resources.

### Placement Rules

- New library code: `source/slavv/<domain>/...`
- New tests:
  - unit: `tests/unit/`
  - integration: `tests/integration/`
  - ui: `tests/ui/`
  - diagnostics/setup: `tests/diagnostic/`
- New developer docs: `docs/`
- New notebook or experiment helpers: `workspace/notebooks/` or `workspace/scripts/`

### Testing Lanes

- Fast lane (default in CI): `pytest -m "unit or integration"`
- Full lane (nightly/manual): `pytest tests/`

Markers are auto-assigned by test folder in `tests/conftest.py`.

## Experiment Workflow

The experiment workflow has been decoupled to allow flexible execution:

1. **Run MATLAB**: `workspace/notebooks/01_Run_Matlab.ipynb` (or CLI) -> `workspace/experiments/...`
2. **Run Python**: `workspace/notebooks/02_Run_Python.ipynb` -> `workspace/experiments/...`
3. **Compare**: `workspace/notebooks/01_End_to_End_Comparison.ipynb` -> `workspace/experiments/...`

Each run captures system information (CPU, RAM, GPU, OS, and software versions) for reproducible performance tracking.

## Key Features

### Modular Architecture

| Module | Responsibility |
|--------|----------------|
| `pipeline.py` | Orchestration, checkpointing |
| `energy.py` | Multi-scale energy field |
| `tracing.py` | Vertex extraction, edge tracing |
| `graph.py` | Network construction |
| `geometry.py` | Cropping, statistics |

### Memory Optimizations

- Sparse adjacency for large-vertex graphs
- Z-slice eigenvalue strategies to reduce peak memory
- Chunking support via `get_chunking_lattice()`

### Algorithm Options

| Parameter | Options |
|-----------|---------|
| `energy_method` | `hessian` (default), `frangi`, `sato` |
| `direction_method` | `hessian`, `uniform` |
| `discrete_tracing` | `False` (continuous), `True` (voxel-snapped) |
| `edge_method` | `tracing` (default), `watershed` |

## Architecture

### High-Level Context

The system is structured as a modular library (`slavv`) used by CLI, Streamlit, and notebook interfaces.

```mermaid
graph TD
    User([User / Researcher])

    subgraph "Interfaces"
        CLI[Command Line Interface]
        WebApp[Streamlit App]
        Jupyter[Jupyter Notebooks]
    end

    subgraph "Core Library (source/slavv)"
        API[SLAVVProcessor]
        IO[IO Utils]
        Algo[Vectorization Logic]
    end

    subgraph "Data Storage"
        FS[File System (.tif, .h5)]
    end

    User --> CLI
    User --> WebApp
    User --> Jupyter
    CLI --> API
    WebApp --> API
    Jupyter --> API
    API --> Algo
    API --> IO
    IO --> FS
```

### Pipeline Data Flow

```mermaid
flowchart LR
    A[Input Volume<br/>(TIFF/Numpy)] --> B{Energy Field<br/>Calculation}
    B -->|Potential Map| C[Vertex Extraction<br/>(Local Maxima)]
    C -->|Seed Points| D[Edge Tracing<br/>(Gradient Descent)]
    D -->|Raw Segments| E[Network Construction<br/>(Graph Logic)]
    E --> F[Vascular Graph<br/>(Nodes + Edges)]
```

### Data Structure Evolution

1. Raster phase (`image`, `energy`): `numpy.ndarray`, shape `(Y, X, Z)`
2. Point phase (`vertices`): dict of arrays, including positions and scales
3. Trace phase (`edges`): list of `(M, 3)` polylines
4. Graph phase (`network`): adjacency-style graph with node/edge attributes

### Design Principles

- Separation of concerns: core logic in `source/slavv`, interfaces outside core modules.
- Stateless processing: pipeline stages pass explicit data dictionaries.
- Explicit configuration: physical parameters come from `params`, defaults are validated.

## Test Roadmap

This roadmap prioritizes fewer, higher-signal tests over broad, repetitive coverage.

### Goals

- Reduce test maintenance cost.
- Improve failure signal quality.
- Keep confidence in critical algorithms and I/O paths.
- Stabilize CI behavior across platforms.

### Success Metrics

- Runtime: reduce default `pytest tests/` wall time by 30-40%.
- Reliability: eliminate non-deterministic temp-path and filesystem failures in CI.
- Signal: each failing test should map to one clear behavior regression.
- Coverage quality: track behavior coverage for core modules instead of raw line count.

### Phases

1. Baseline and guardrails:
   - Record runtime + slowest tests.
   - Keep marker taxonomy (`unit`, `integration`, `ui`, `slow`, `regression`).
   - Maintain CI split (fast lane + full lane).
2. Remove low-value redundancy:
   - Collapse repetitive tests into scenario-driven parameterized tests.
   - Remove implementation-detail assertions with low regression value.
3. Stabilize environment-sensitive tests:
   - Keep temp file handling centralized in fixtures.
   - Separate product regressions from environment/setup failures.
4. Strengthen high-value regression coverage:
   - Focus on core tracing/energy outputs and I/O round-trips.
   - Maintain deterministic fixture-based datasets.
5. Improve developer feedback:
   - Keep fast local lane and full nightly/manual lane.
   - Track runtime/flakiness trend in CI artifacts.

### Immediate Next Steps

1. Audit top 5 slowest modules with `pytest --durations=10`.
2. Continue refactoring oversized test modules to scenario-based tests.
3. Add failure triage labels in CI output (`behavior`, `test defect`, `infra`).

## Architecture Decisions

ADRs capture significant technical decisions with context, decision, and consequences.

### ADR-0001 (Accepted, 2026-01-27)

- Title: Use Python over MATLAB for data processing.
- Decision: Python is the primary implementation; MATLAB remains reference-only.
- Reasoning: better maintainability, faster iteration, and measured runtime improvements.
- Consequence: prioritize parity checks and migration of remaining MATLAB-only features in Python.

## Migration Summary

Parity labels:

- `Exact`: intended and validated equivalent behavior
- `Approximate`: functional but not numerically identical in all cases
- `Omitted`: not implemented yet

### Core MATLAB -> Python mapping

| MATLAB | Python | Parity |
|---|---|---|
| `vectorize_V200.m` | `source/slavv/pipeline.py:SLAVVProcessor.process_image` | Approximate |
| `get_energy_V202.m` | `source/slavv/energy.py:calculate_energy_field` | Approximate |
| `get_vertices_V200.m` | `source/slavv/tracing.py:extract_vertices` | Approximate |
| `get_edges_V300.m` | `source/slavv/tracing.py:extract_edges` | Approximate |
| `get_network_V190.m` | `source/slavv/graph.py:construct_network` | Approximate |

### Migration priorities

1. Keep behavior stable with regression tests before performance tuning.
2. Prioritize parity for energy, tracing, and network construction.
3. Validate all export/import round-trips (`MAT`, `CASX`, `VMV`, `CSV`, `JSON`).

## See Also

- [ARCHIVE.md](ARCHIVE.md) - Historical reports and post-mortems
- [WORKSPACE.md](WORKSPACE.md) - Workspace scripts, notebooks, and experiments guide
