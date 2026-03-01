# Development Guide

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core Pipeline (`pipeline.py`) | Complete | Energy -> Vertices -> Edges -> Network |
| I/O (`io_utils.py`) | Complete | MAT, CASX, VMV, CSV, JSON, DICOM, TIFF |
| Visualization (`visualization.py`) | Complete | 2D/3D Plotly, animations |
| ML Curation (`ml_curator.py`) | Functional | Logistic/RF classifiers |
| Test Suite | Active | Unit + integration coverage |

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

## Future Work

### High Priority

- [ ] Numba JIT for edge tracing
- [ ] Parallel chunk processing with `joblib`
- [x] CLI tool via `pyproject.toml` entry points

### Medium Priority

- [ ] Interactive curation GUI in Streamlit
- [ ] Coordinate system validation
- [ ] Kernel fidelity improvements

### Low Priority

- [ ] API reference (Sphinx/MkDocs)
- [x] Jupyter naming convention (`00_`, `01_`, etc.)
- [ ] Jupyter tutorial notebooks (`07_Tutorial.ipynb`)
- [ ] Full `mypy` compliance

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design diagrams
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - MATLAB to Python mapping
