# Development Guide

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core Pipeline (`pipeline.py`) | ✅ Complete | Energy → Vertices → Edges → Network |
| I/O (`io_utils.py`) | ✅ Complete | MAT, CASX, VMV, CSV, JSON, DICOM, TIFF |
| Visualization (`visualization.py`) | ✅ Complete | 2D/3D Plotly, animations |
| ML Curation (`ml_curator.py`) | ✅ Functional | Logistic/RF classifiers |
| Test Suite | ✅ 81 tests | 38 files, unit + integration |

---

## Experiment Workflow
The experiment workflow has been decoupled to allow flexible execution:
1. **Run MATLAB**: `notebooks/01_Run_Matlab.ipynb` (or CLI) $\rightarrow$ `experiments/XX_matlab_run/`
2. **Run Python**: `notebooks/01_Run_Python.ipynb` $\rightarrow$ `experiments/XX_python_run/`
3. **Compare**: `notebooks/01_End_to_End_Comparison.ipynb` $\rightarrow$ `experiments/XX_comparison_report/`

**System Information Capture**: Each run automatically logs CPU, RAM, GPU, OS, and software versions for performance tracking across different machines.


---

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
- **Sparse adjacency**: 37GB → ~1MB for 200k vertices
- **Z-slice eigenvalues**: 200x peak memory reduction
- **Chunking**: Large volumes via `get_chunking_lattice()`

### Algorithm Options
| Parameter | Options |
|-----------|---------|
| `energy_method` | `hessian` (default), `frangi`, `sato` |
| `direction_method` | `hessian`, `uniform` |
| `discrete_tracing` | `False` (continuous), `True` (voxel-snapped) |
| `edge_method` | `tracing` (default), `watershed` (20x faster) |

---

## Performance Guide

### Quick Wins

| Optimization | Impact | How |
|--------------|--------|-----|
| GPU acceleration | 10-100x | `use_gpu=True` (requires CuPy) |
| Discrete tracing | 2x | `discrete_tracing=True` |
| Watershed edges | 20x | `edge_method='watershed'` |
| FFT convolution | 10x | Auto for σ ≥ 10 |

### Memory Tips
- Enable `memory_map=True` in `load_tiff_volume()`
- Reduce `max_voxels_per_node_energy` for large volumes

---

## Future Work

### High Priority
- [ ] Numba JIT for edge tracing (~100x speedup)
- [ ] Parallel chunk processing with `joblib`
- [x] CLI tool via `pyproject.toml` entry points

### Medium Priority
- [ ] Interactive curation GUI in Streamlit
- [ ] Coordinate system validation
- [ ] Kernel fidelity improvements

### Visualization & Export Enhancements (Recommended)
Building on work with VessMorphoVis/Blender exports:
- [ ] **Pre-render 3D views automatically**: Generate PNG snapshots from multiple angles during the pipeline run.
- [ ] **Add interactive 3D viewer**: Create a web-based viewer (e.g., using three.js or pyvista) so Blender isn't strictly required.
- [ ] **Generate animated rotations**: Create MP4/GIFs of the rotating network.
- [ ] **Create comparison gallery**: Side-by-side renders of MATLAB vs Python outputs.

### Low Priority
- [ ] API reference (Sphinx/MkDocs)
- [x] Jupyter naming convention (`00_`, `01_`, etc.)
- [ ] Jupyter tutorial notebooks (`07_Tutorial.ipynb`)
- [ ] Full `mypy` compliance

---

## See Also
- [ARCHITECTURE.md](../reference/ARCHITECTURE.md) - System design diagrams
- [MIGRATION_GUIDE.md](../reference/MIGRATION_GUIDE.md) - MATLAB→Python mapping
