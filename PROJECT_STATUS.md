# SLAVV Python — Project Status

**Date:** 2026-05-19  
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
| Package Python files (`slavv_python/`) | **178** |
| Test Python files (`tests/`) | **88** |
| Package lines of code | **~26,300** |
| Test lines of code | **~8,500** |
| Core dependencies | 14 (NumPy, SciPy, scikit-image, NetworkX, Plotly, etc.) |
| Optional extras | 10 (app, ml, notebooks, dicom, sitk, cupy, zarr, napari, accel, tui) |
| CLI entrypoints | `slavv` (argparse), `slavv-app` (Streamlit launcher) |
| Pipeline profiles | `paper` (default), `matlab_compat` |

---

## 3. Repository Structure

```
slavv2python/
├── slavv_python/                   # Main package (178 .py files)
│   ├── engine/                     # Pipeline orchestration, run context, lifecycle
│   │   └── state/                  # Run tracking, models, snapshots, resume policy
│   ├── processing/                 # Scientific computation
│   │   ├── image/                  # Normalization, tiling
│   │   └── stages/                 # Core pipeline stages (ACTUAL code lives here)
│   │       ├── energy/             # Hessian response, gradients, backends, chunking
│   │       ├── vertices/           # Vertex extraction, painting, selection
│   │       ├── edges/              # Watershed, tracing, selection, cleanup, bridge insertion
│   │       │   └── matlab_algorithms/  # MATLAB-shaped parity shims
│   │       └── network/            # Strand assembly, graph construction
│   ├── analytics/                  # Analysis & metrics
│   │   ├── parity/                 # MATLAB exact proof harness (15 modules, ~170 KB)
│   │   ├── curation/               # Automated & ML curators
│   │   └── metrics/                # Intensity, topology metrics
│   ├── storage/                    # Data I/O
│   │   ├── loaders/                # TIFF, network loaders
│   │   └── exporters/              # JSON v1 exporter
│   ├── interface/                  # User-facing surfaces
│   │   ├── cli/                    # argparse CLI (13 modules + TUI)
│   │   ├── streamlit/              # Streamlit web app (6 pages)
│   │   ├── shared_services/        # Cross-UI service layer
│   │   └── shared_state/           # Cross-UI state management
│   ├── visualization/              # Plotting & rendering
│   │   └── network_plots/          # 2D, 3D, statistical, dashboard plots
│   ├── schema/                     # Data models (results schema)
│   ├── workflows/                  # Pipeline orchestration helpers, profiles
│   │   └── pipeline/               # Session, execution, resolution, artifacts
│   └── utils/                      # Validation, math, formatting, system info
│
├── tests/                          # Test suite (88 .py files)
│   ├── unit/                       # By-owner unit tests (analysis, apps, core, io, etc.)
│   ├── integration/                # End-to-end, paper profile, public API tests
│   │   └── parity/                 # Parity-specific integration tests
│   ├── ui/                         # Streamlit/visualization tests (8 files)
│   ├── runtime/                    # Run-state management tests
│   ├── support/                    # Shared test builders & fixtures
│   └── fixtures/                   # Test data fixtures
│
├── scripts/                        # Developer scripts
│   ├── cli/                        # parity_experiment.py, execution trace comparison
│   └── diagnostics/                # MATLAB artifact inspection scripts (7 files)
│
├── docs/                           # Documentation
│   ├── reference/                  # Maintained technical references
│   │   ├── core/                   # Parity plan, mapping, proof findings, energy methods
│   │   ├── workflow/               # Paper profile, naming guide, extraction guide
│   │   ├── backends/               # Backend documentation
│   │   └── papers/                 # Published paper PDF
│   └── investigations/             # Archival investigation narratives
│       ├── v22-pointer-corruption/ # Historical pointer bug analysis
│       └── translation_pair_analysis/ # Missing/extra pair deep-dive
│
├── workspace/                      # Developer experiment workspace
│   ├── oracles/                    # Preserved MATLAB oracle vectors
│   ├── runs/                       # Experiment trial runs
│   ├── reports/                    # Promoted proof summaries
│   ├── datasets/                   # Test datasets
│   └── scratch/                    # Temporary scratch files
│
├── external/                       # Vendored external dependencies
│   ├── Vectorization-Public/       # Canonical MATLAB source (git submodule)
│   ├── blender_resources/          # Blender visualization assets
│   └── neurovasc-db/               # Neurovascular database
│
├── pyproject.toml                  # Build config, tooling (ruff, mypy, pytest)
├── AGENTS.md                       # AI agent instructions
├── CHANGELOG.md                    # Release history
├── README.md                       # User-facing README
└── docs/ROADMAP.md                 # Developer command center & task tracking
```

---

## 4. Pipeline Stages & Parity Status

The pipeline runs four sequential stages. Each has a separate public-workflow status and an exact-parity-proof status:

| # | Stage | Description | Public Workflow | Exact Parity | Proof Detail |
|:--|:------|:------------|:---------------:|:------------:|:-------------|
| 1 | **Energy** | Multi-scale Hessian matched filtering to compute voxel-level vessel energy | ✅ Complete | ✅ Complete | Native `python_native_hessian` is the canonical exact-compatible provenance |
| 2 | **Vertices** | Local energy minima extraction & painting | ✅ Complete | ✅ Verified | Successfully certified for downstream stages |
| 3 | **Edges** | Global watershed candidate generation → selection → cleanup → bridge insertion | ✅ Complete | ⚠️ 88.7% | 1,062 / 1,197 MATLAB oracle pairs matched; 135 missing, 371 over-generated |
| 4 | **Network** | Strand assembly from curated edges into a connected graph | ✅ Complete | ⏳ Pending | Blocked on upstream edge closure |

### Key Parity Milestones Reached
- **v29 breakthrough** (May 2026): Parameter alignment (`edge_number_tolerance` 2→4) + NaN stability fix → jumped from 80.0% to **88.7%** match rate
- **Trace order fix** (May 2026): Seeded RNG for conflict painting → jumped from 14% to 56% → then to 80% with frontier ordering fixes
- **Native energy cutover**: Python Hessian matched filtering is now the canonical energy implementation; MATLAB energy import is retired from the exact route

### Remaining Blockers for 100% Exact Parity
1. **Frontier ordering divergence** — Fine-grained seed selection priority differences in the global watershed crawler
2. **Hub vertex complexity** — Branching decisions near high-degree junction clusters
3. **Candidate filtering alignment** — Python acceptance criteria not yet tightened to match MATLAB `get_edges_by_watershed` filters
4. **Downstream cascade** — Bridge insertion, network assembly, and strand ordering cannot be proven until edges close

---

## 5. Public Workflow Status

| Surface | Status | Notes |
|:--------|:------:|:------|
| `slavv run --profile paper` | 🟡 Not fully verified | Pipeline code exists; end-to-end verification not tracked |
| `slavv run --profile matlab_compat` | 🟡 Not fully verified | Available but not prioritized |
| `slavv analyze` | 🟡 Not fully verified | Consumes `network.json` |
| `slavv plot` | 🟡 Not fully verified | Generates HTML plots |
| `slavv-app` (Streamlit) | 🟡 Not fully verified | Multi-page app with processing, dashboard, curation, visualization |
| `network.json` export | ✅ Implemented | Versioned JSON v1 exporter exists |
| Integration test coverage | 🟡 Partial | 6 integration tests exist; full paper-profile e2e test tracked but not confirmed green |

> **[PAPER-001]** in the roadmap tracks public workflow health as "NOT TRACKED YET" — this is a gap that needs attention.

---

## 6. Architecture Health

### ✅ Strengths
- **Clean modular package layout** — Code is well-organized into `engine`, `processing`, `analytics`, `storage`, `interface`, `visualization`, `workflows`, `utils`, and `schema` subpackages
- **Real import paths work** — All internal imports use the new `slavv_python.processing.stages.*` and `slavv_python.engine.*` paths consistently
- **Pipeline profiles** — Clean separation of `paper` vs `matlab_compat` parameter presets
- **Run state management** — Structured `run_dir` metadata with snapshot, resume, and progress tracking (11 modules in `engine/state/`)
- **Parity infrastructure** — Comprehensive proof harness with oracle promotion, candidate capture, gap analysis, and fail-fast tooling (15 modules in `analytics/parity/`)
- **Quality tooling** — Ruff (lint + format), mypy (gradual typing), pytest (with markers), pre-commit hooks all configured

### ⚠️ Concerns

#### ~~Documentation / Code Path Mismatch~~ ✅ FIXED
All documentation has been updated to reference the actual current paths.

#### ~~pyproject.toml Entrypoint Mismatch~~ ✅ FIXED
```toml
# pyproject.toml now correctly says:
slavv-app = "slavv_python.interface.streamlit_launcher:main"
slavv = "slavv_python.interface.cli:main"
```

#### ~~mypy Configuration References Stale Paths~~ ✅ FIXED
The `[tool.mypy]` `files` list has been updated to reference the actual module paths.

#### Package Import Failure (MEDIUM)
`import slavv_python` fails in the current environment. The `__init__.py` references `from slavv_python.core import SlavvPipeline` indirectly through `from .engine import SlavvPipeline`, which should work, but:
- The CUDA/setuptools warnings suggest environment configuration issues
- The `slavv_python/storage/loaders/tiff.py` and `slavv_python/storage/loaders/network.py` imports at package root may fail if heavy dependencies aren't installed

#### ~~Root Directory Clutter~~ ✅ FIXED
All stale files (`all_night_parity.log`, `all_night_suite_stdout.log`, `report-v27.txt`, `trace-v2.json`, `trace-v3.json`, `fix_parity_imports.py`, `run_all_night_suite.py`, `index.jsonl`) have been moved to `workspace/scratch/`.

---

## 7. Test Coverage Overview

| Category | Location | File Count | Key Areas |
|:---------|:---------|:----------:|:----------|
| Unit | `tests/unit/` | ~65 | Analysis, apps, core processing, I/O, models, runtime, utils, visualization, workflows, scripts |
| Integration | `tests/integration/` | 7 | End-to-end pipeline, paper profile, public API, curator, parity |
| UI | `tests/ui/` | 8 | Streamlit dashboard, edge coloring, network slicing, visualization exports |
| Runtime | `tests/runtime/` | ~5 | Run-state management |
| Support | `tests/support/` | ~5 | Shared test builders and synthetic fixtures |

Test markers: `unit`, `integration`, `ui`, `diagnostic`, `slow`, `regression`

---

## 8. Quality Gate Configuration

| Tool | Status | Scope |
|:-----|:------:|:------|
| **Ruff lint** | Configured | `slavv_python/`, `tests/` — 14 rule sets enabled |
| **Ruff format** | Configured | Double-quote, space-indent, docstring formatting |
| **mypy** | Configured (⚠️ stale paths) | Gradual typing, `follow_imports = "skip"` |
| **pytest** | Configured | `tests/` directory, 6 markers, deprecation filters |
| **pre-commit** | Configured | Hooks installed |

---

## 9. Roadmap Summary

### Critical Priority
- **PARITY-002**: Achieve and hold ≥80% edge match rate ✅ (currently 88.7%)
  - Measure 2 (Frontier insertion): ✅ COMPLETED
  - Measure 3 (Candidate filtering): NOT STARTED — next up
  - Boundary conditions: NOT STARTED

### High Priority
- **PARITY-003**: Achieve 100% exact parity — BLOCKED on PARITY-002 edge closure
- **PAPER-001**: Public paper workflow health — NOT TRACKED YET

### Medium Priority
- **PERF-001**: Algorithm performance optimization — PAUSED (frontier O(N²)→O(log N) may have introduced ordering bugs)

### Low Priority
- **INVEST-005**: Execution trace comparison — SUPERSEDED by whole-frontier algorithmic fixes

---

## 10. Immediate Action Items

### ~~Critical~~ ✅ COMPLETED
1. ~~**Fix documentation/code path mismatch**~~ — Updated all docs to reference actual paths
2. ~~**Fix pyproject.toml entrypoints**~~ — CLI and Streamlit launcher paths now use `slavv_python.interface.*`
3. ~~**Fix mypy file paths**~~ — `[tool.mypy]` now references actual module locations

### High
4. **Verify `slavv run` end-to-end** — PAPER-001 is not tracked; need to confirm the public workflow actually works
5. **Clean up root directory** — Move stale logs, traces, and scratch scripts to `workspace/scratch/`

### Medium
6. **Start Measure 3** — Candidate filtering alignment is the next parity work item (now that Measure 2 is complete and 88.7% is baseline)
7. **Resolve package import** — Ensure `import slavv_python` works cleanly in the target Python 3.11+ environment

---

## 11. Technology Stack

| Layer | Technology |
|:------|:-----------|
| Language | Python ≥3.11 |
| Scientific | NumPy ≥2.0, SciPy ≥1.10, scikit-image ≥0.21, scikit-learn ≥1.2 |
| Graph | NetworkX ≥2.6 |
| I/O | tifffile, h5py, joblib, Pillow |
| Visualization | Matplotlib ≥3.3, Plotly ≥5.0, Seaborn ≥0.9 |
| Web App | Streamlit ≥1.56 |
| CLI | argparse (standard library) |
| Data | Pandas ≥3.0 |
| Quality | Ruff ≥0.4, mypy ≥1.8, pytest ≥7.0, pre-commit ≥3.0 |
| Build | setuptools ≥61, wheel |
