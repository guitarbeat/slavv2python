# SLAVV2Python Development Roadmap

This document outlines the remaining work to bring `slavv2python` to full production-ready status with MATLAB parity and beyond.

## Current Status ✅

| Component | Status | Notes |
|---|---|---|
| Core Pipeline (`pipeline.py`) | **Functional** | Energy → Vertices → Edges → Network complete |
| I/O (`io_utils.py`) | **Complete** | MAT, CASX, VMV, CSV, JSON, DICOM, TIFF |
| Visualization (`visualization.py`) | **Complete** | 2D/3D Plotly, animations, export |
| ML Curation (`ml_curator.py`) | **Functional** | Logistic/RF classifiers, feature extraction |
| Test Suite | **81 tests** | Covers core pipeline, I/O, visualization |
| Documentation | **Complete** | Mapping, Innovations, Performance docs |
| Checkpointing | **Built-in** | `checkpoint_dir` argument for resume |

---

## Phase 1: Performance (Priority: High) 🚀

### 1.1 Numba JIT for Edge Tracing
- **File:** `tracing.py` → `trace_edge()`
- **Status:** ⚠️ Temporarily disabled due to dtype issues with Python 3.7
- **Impact:** ~100x speedup for tracing loops
- **Effort:** Medium (requires Numba version/dtype fix)

### 1.2 Memory-Efficient Eigenvalue Computation ✅
- **File:** `energy.py` → `calculate_energy_field()`
- **Status:** ✅ **COMPLETE** - Z-slice batched eigenvalue computation
- **Impact:** Reduced peak memory from 1.3GB to ~6MB per slice
- **Effort:** Done!

### 1.3 Sparse Adjacency Matrix ✅
- **File:** `graph.py` → `construct_network()`
- **Status:** ✅ **COMPLETE** - Replaced dense boolean matrix with adjacency list
- **Impact:** Reduced memory from 37GB (!) to ~few MB for 200k vertices
- **Effort:** Done!

### 1.3 FFT Convolution for Large σ
- **File:** `energy.py` → `calculate_energy_field()`
- **Rule:** If σ ≥ 10, use `scipy.signal.fftconvolve`
- **Impact:** ~10x speedup for multi-scale energy
- **Effort:** Low

### 1.3 Parallel Chunk Processing
- **File:** `utils.py` → `get_chunking_lattice()`
- **Tool:** `joblib.Parallel` for CPU parallelization
- **Impact:** Linear speedup with cores
- **Effort:** Low

### 1.4 GPU Acceleration (Optional)
- **Library:** CuPy as drop-in NumPy replacement
- **Flag:** Add `use_gpu=True` to `SLAVVProcessor`
- **Impact:** 10-100x for energy/Hessian
- **Effort:** High (requires testing infrastructure)

---

## Phase 2: Accuracy & Parity (Priority: Medium) 🎯

### 2.1 Kernel Fidelity
- **Gap:** MATLAB PSF kernels are more detailed
- **Action:** Port exact `get_filter_kernel.m` logic
- **Effort:** Medium

### 2.2 Coordinate System Validation
- **Gap:** Python uses `(y, x, z)`; verify all transforms
- **Action:** End-to-end coordinate audit with known geometry
- **Effort:** Low

### 2.3 Discrete Tracing Mode Validation
- **Gap:** `discrete_tracing=True` exists but needs MATLAB parity test
- **Action:** Compare outputs on identical synthetic data
- **Effort:** Low

---

## Phase 3: User Experience (Priority: Medium) 🖥️

### 3.1 Interactive Curation GUI
- **Gap:** MATLAB has graphical vertex/edge curator
- **Action:** Streamlit widget for manual corrections
- **Effort:** High

### 3.2 CLI Tool
- **Gap:** All execution requires Python script
- **Action:** Add `slavv` CLI via `pyproject.toml` entry points
- **Effort:** Low

### 3.3 Progress Logging Improvements
- **Gap:** Current logging is sparse during long runs
- **Action:** Add ETA estimation, memory usage reporting
- **Effort:** Low

---

## Phase 4: Testing & CI (Priority: Low) 🧪

### 4.1 Expand Test Coverage
- **Current:** 38 test files, 81 tests
- **Gap:** No integration tests with real data
- **Action:** Add MATLAB output comparison tests

### 4.2 Benchmark Suite
- **Gap:** No automated performance tracking
- **Action:** Add pytest-benchmark for regression detection

### 4.3 Type Hints & Linting
- **Gap:** Partial type coverage
- **Action:** Full `mypy` compliance

---

## Phase 5: Documentation & Publication (Priority: Low) 📚

### 5.1 API Reference
- **Gap:** No auto-generated API docs
- **Action:** Set up Sphinx or MkDocs

### 5.2 Tutorial Notebooks
- **Gap:** No Jupyter examples
- **Action:** Create "Getting Started" notebook

### 5.3 Paper Supplement
- **Files:** `docs/INNOVATIONS_AND_MODIFICATIONS.md`
- **Action:** Polish for journal submission

---

## Unmapped MATLAB Files (43/152)

These files were intentionally **not** ported:
- Example scripts: `vectorization_script_*`, `noise_sensitivity_*`
- Legacy helpers: `histogram_plotter.m`, `paint_vertex_image.m`
- Format converters: `partition_casx_by_xy_bins.m`

---

## Quick Wins (Do Next)

1. ✅ **Checkpointing** — Done!
2. ✅ **DRY Refactoring** — Done!
3. 🔲 **Numba JIT** — Highest ROI performance fix
4. 🔲 **CLI Tool** — Easy UX improvement
5. 🔲 **FFT Convolution** — Low-effort, high-impact
