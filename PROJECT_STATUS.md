# SLAVV Python — Project Status

**Date:** 2026-05-20  
**Version:** 0.1.0 (Beta)  
**Python:** ≥3.11  
**License:** GPL-3.0

---

## 1. What Is This Project?

SLAVV (Segmentation-Less, Automated, Vascular Vectorization) is a Python port of a published MATLAB algorithm for extracting 3D vascular networks from microscopy volumes. The pipeline takes a TIFF volume as input, computes multi-scale energy fields, extracts vertices and edges via a global watershed, assembles a network graph, and exports an authoritative `network.json` for downstream analysis and visualization.

The project has two parallel goals:
1. **Public paper workflow** — A standalone native Python TIFF-to-network pipeline that users can run without MATLAB.
2. **Exact MATLAB parity** — A developer proof track that mathematically validates the Python output against preserved MATLAB oracle vectors.

---

## 2. Codebase At a Glance

| Metric | Count |
|:---|---:|
| Package Python files (`slavv_python/`) | **183** |
| Test Python files (`tests/`) | **88** |
| Package lines of code | **~27,500** |
| Test lines of code | **~8,500** |
| Core dependencies | 15 (NumPy, SciPy, scikit-image, NetworkX, Plotly, etc.) |
| Optional extras | 10 (app, ml, notebooks, dicom, sitk, cupy, zarr, napari, accel, tui) |
| CLI entrypoints | `slavv` (argparse), `slavv-app` (Streamlit launcher) |
| Pipeline profiles | `paper` (default), `matlab_compat` |

---

## 3. Repository Structure

```
slavv2python/
├── slavv_python/                   # Main package (183 .py files)
│   ├── engine/                     # Pipeline orchestration, run context, lifecycle
│   │   └── state/                  # Run tracking, models, snapshots, resume policy
│   ├── processing/                 # Scientific computation
│   │   ├── image/                  # Normalization, tiling
│   │   └── stages/                 # Core pipeline stages
│   │       ├── energy/             # Hessian response, gradients, backends, chunking
│   │       ├── vertices/           # Vertex extraction, painting, selection
│   │       ├── edges/              # Watershed, tracing, selection, cleanup, bridge insertion
│   │       └── network/            # Strand assembly, graph construction
│   ├── analytics/                  # Analysis & metrics
│   │   ├── parity/                 # MATLAB exact proof harness
│   │   ├── curation/               # Automated & ML curators
│   │   └── metrics/                # Intensity, topology metrics
│   ├── storage/                    # Data I/O
│   │   ├── loaders/                # TIFF, network loaders
│   │   └── exporters/              # JSON v1 exporter
│   ├── interface/                  # User-facing surfaces
│   │   ├── cli/                    # argparse CLI
│   │   ├── streamlit/              # Streamlit web app
│   │   └── shared_services/        # Cross-UI service layer
│   ├── visualization/              # Plotting & rendering
│   │   └── network_plots/          # 2D, 3D, statistical, dashboard plots
│   ├── schema/                     # Data models (results schema)
│   ├── workflows/                  # Pipeline orchestration helpers, profiles
│   └── utils/                      # Validation, math, formatting, system info
│
├── tests/                          # Test suite (88 .py files)
│   ├── unit/                       # By-owner unit tests (361/378 passing)
│   ├── integration/                # End-to-end, paper profile, public API tests
│   │   └── parity/                 # Parity-specific integration tests
│   ├── ui/                         # Streamlit/visualization tests
│   ├── runtime/                    # Run-state management tests
│   └── support/                        # Shared test builders & fixtures
│
├── scripts/                        # Developer scripts
│   ├── cli/                        # parity_experiment.py, execution trace comparison
│   └── diagnostics/                # MATLAB artifact inspection scripts
│
├── docs/                           # Documentation
│   ├── reference/                  # Maintained technical references
│   └── investigations/             # Archival investigation narratives
│
├── workspace/                      # Developer experiment workspace
│   ├── oracles/                    # Preserved MATLAB oracle vectors
│   ├── runs/                       # Experiment trial runs
│   └── scratch/                    # Temporary scratch files
│
└── external/                       # Vendored external dependencies
    ├── Vectorization-Public/       # Canonical MATLAB source (git submodule)
    └── neurovasc-db/               # Neurovascular database (git-annex)
```

---

## 4. Pipeline Stages & Parity Status

The pipeline runs four sequential stages. Each has a separate public-workflow status and an exact-parity-proof status:

| # | Stage | Description | Public Workflow | Exact Parity | Proof Detail |
|:--|:------|:------------|:---------------:|:------------:|:-------------|
| 1 | **Energy** | Multi-scale Hessian matched filtering | ✅ Complete | ✅ Complete | Native `python_native_hessian` is bit-accurate |
| 2 | **Vertices** | Local minima extraction & painting | ✅ Complete | ✅ Verified | Successfully certified for downstream |
| 3 | **Edges** | Global watershed → tracing → selection | ✅ Complete | ⚠️ 88.7% | 1,062 / 1,197 pairs matched (v32 Precision Fix) |
| 4 | **Network** | Graph assembly & strand smoothing | ✅ Complete | ⏳ Pending | Verified end-to-end; proof pending edge closure |

### Key Parity Milestones Reached
- **v32 Precision Fix** (May 2026): Forced core watershed to `float64` and stabilized frontier ordering to match MATLAB `double` tie-breaking.
- **v29 breakthrough** (May 2026): Parameter alignment (`edge_number_tolerance` 2→4) + NaN stability fix → jumped to **88.7%** match rate.
- **Native energy cutover**: Python Hessian matched filtering is now the canonical energy implementation.

### Remaining Blockers for 100% Exact Parity
1. **Hub vertex complexity** — Branching decisions near high-degree junction clusters (requires Linear-Index tie-break).
2. **git-annex data retrieval** — Required for multi-dataset stability testing beyond the current oracle.
3. **Downstream cascade** — Final proof of bridge insertion and strand ordering.

---

## 5. Public Workflow Status

| Surface | Status | Notes |
|:--------|:------:|:------|
| `slavv run --profile paper` | ✅ Verified | End-to-end verified with synthetic TIFF (May 2026) |
| `slavv analyze` | ✅ Verified | Topology summary and statistics verified |
| `slavv plot` | ✅ Verified | Interactive HTML Plotly dashboard verified |
| `slavv-app` (Streamlit) | ✅ Verified | All analytical dependencies resolved; server starts on 8501 |
| `network.json` export | ✅ Verified | Authoritative JSON v1 round-trip verified |
| Integration test coverage | 🟡 Partial | 96% unit test pass rate; integration suite needs final polish |

---

## 6. Architecture Health

### ✅ Strengths
- **End-to-End Viability** — Milestone **PAPER-001** is reached. The product works out-of-the-box.
- **Precision Alignment** — Core computational buffers moved to `float64` to match MATLAB's numerical precision.
- **CI/CD infrastructure** — GitHub Actions automated regression gate added for all main branch commits.
- **Package Integrity** — Systemic `ImportError`s across analytics, storage, and interface resolved.

### ⚠️ Concerns

#### ML Component Gaps (MEDIUM)
Several machine-learning curation components were found to be missing from the codebase and have been **stubbed** to allow the Streamlit app to function. A proper re-implementation or restoration is required for the ML curation track.

#### git-annex Missing (LOW)
The current development environment lacks `git-annex`, blocking the retrieval of real multi-gigabyte dataset volumes for comprehensive stability testing.

---

## 7. Quality Gate Status

| Tool | Status | Pass Rate / Detail |
|:-----|:------:|:-------------------|
| **Ruff** | ✅ Configured | Format and Lint enforcement in CI |
| **mypy** | ✅ Configured | Type safety checks in CI |
| **pytest** | ✅ Configured | **361 / 378 (96%) passing** |
| **CI/CD** | ✅ Active | GitHub Actions `regression-gate.yml` |

---

## 8. Immediate Action Items

### High
1. **Linear-Index Tie-Breaking** — Implement secondary sort key in watershed to bridge the final 11.3% parity gap.
2. **Restore ML Components** — Replace stubs in `analytics.curation.machine_learning` with real logic or restored modules.

### Medium
3. **Retrieve Real Data** — Execute `git annex get external/` in a compatible environment for multi-dataset testing.
4. **Update Integration Tests** — Refine the integration suite to match the new `float64` precision expectations.

---

## 9. Technology Stack

| Layer | Technology |
|:------|:-----------|
| Language | Python ≥3.11 |
| Scientific | NumPy ≥2.0, SciPy ≥1.10, scikit-image ≥0.21, scikit-learn ≥1.2 |
| Graph | NetworkX ≥2.6 |
| I/O | tifffile, h5py, joblib, zarr |
| Visualization | Plotly ≥5.0, Streamlit ≥1.56 |
| CI/CD | GitHub Actions |
