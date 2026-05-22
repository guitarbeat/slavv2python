# SLAVV Python Developer Command Center & Roadmap

**Date:** 2026-05-20  
**Version:** 0.1.0 (Beta)  
**Python:** ≥3.11  
**License:** GPL-3.0

This document serves as the central command center, official roadmap, engineering objectives, and project status for the **SLAVV (Strand Localization and Vessel Vectorization)** Python codebase.

## Navigation & Architecture

| Link | Purpose |
| :--- | :--- |
| [Live Proof Status](reference/core/EXACT_PROOF_FINDINGS.md) | Current v22 readouts and regression failures |
| [Investigation Findings](investigations/translation_pair_analysis/INVESTIGATION_FINDINGS.md) | Deep analysis of missing/extra translation pairs |
| [Policy & Implementation Phases](reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md) | Canonical claim boundaries and long-term roadmap |
| [Parity Mapping](reference/core/MATLAB_PARITY_MAPPING.md) | Module-for-module structure against MATLAB |

### What Is This Project?
SLAVV is a Python port of a published MATLAB algorithm for extracting 3D vascular networks from microscopy volumes. The pipeline takes a TIFF volume as input, computes multi-scale energy fields, extracts vertices and edges via a global watershed, assembles a network graph, and exports an authoritative `network.json` for downstream analysis and visualization.

The project has two parallel goals:
1. **Public paper workflow** — A standalone native Python TIFF-to-network pipeline that users can run without MATLAB.
2. **Exact MATLAB parity** — A developer proof track that mathematically validates the Python output against preserved MATLAB oracle vectors.

## Codebase & Pipeline Status

| Metric | Count |
|:---|---:|
| Package Python files | **183** |
| Test Python files | **88** |
| Package lines of code | **~27,500** |
| Test lines of code | **~8,500** |

**Pipeline Stages & Parity Status**
| # | Stage | Description | Public Workflow | Exact Parity | Proof Detail |
|:--|:------|:------------|:---------------:|:------------:|:-------------|
| 1 | **Energy** | Multi-scale Hessian matched filtering | ✅ Complete | ✅ Complete | Native `python_native_hessian` is bit-accurate |
| 2 | **Vertices** | Local minima extraction & painting | ✅ Complete | ✅ Verified | Successfully certified for downstream |
| 3 | **Edges** | Global watershed → tracing → selection | ✅ Complete | ⚠️ 88.7% | 1,062 / 1,197 pairs matched (v32 Precision Fix) |
| 4 | **Network** | Graph assembly & strand smoothing | ✅ Complete | ⏳ Pending | Verified end-to-end; proof pending edge closure |

**Public Workflow Verified:** `slavv run`, `slavv analyze`, `slavv plot`, and `slavv-app` (Streamlit) are all verified end-to-end as of May 21, 2026.

## Active Priority Queue & TODOs

> **Core Strategy:** The parity track is impressive engineering (14% → 88.7%), but the public product priority must remain stable. We focus on a balanced approach: Maintain Public Workflow Stability (Critical) while pursuing Parallel Active Parity Track (High).

### 🔴 Priority 1: Prove the Product Works (PAPER-001)
**Status: COMPLETE (Verified & Locked 2026-05-21)**

- [x] **End-to-End Pipeline Execution:** Verify `slavv run` completes end-to-end and writes structured run directories.
- [x] **Downstream Consumers:** Verify `slavv analyze` and `slavv plot` are fully functional.
- [x] **Interactive UI:** Verify `slavv-app` successfully starts on port 8501.
- [x] **Import Resolution:** Resolved systemic `ImportError`s across analytics, storage, and interface packages.
- [x] **Integration Testing:** Write a paper-profile integration test (`test_paper_profile.py`).
- [x] Add integration test covering the native `paper` profile pipeline (TIFF-to-network-to-export) in the CI/CD regression gate.

### 🟡 Priority 2: Stabilize & Document What We Have
**Status: COMPLETE (100% Test Pass Rate)**

- [x] **Verify test suite passes** — 100% test pass rate achieved (340 unit tests, 13 integration tests passing).
- [x] **Verify quality gate passes** — Linting (Ruff), Type Checking (Mypy), and Tests (Pytest) are all green.
- [x] **Clean up CHANGELOG.md** — Documented stabilization work and typed object migration.
- [x] **Confirm `pyproject.toml` entrypoints resolve**
- [x] **CI/CD** — Added GitHub Actions workflow (`.github/workflows/regression-gate.yml`) with synthetic integration tests.
- [x] **Documentation** — Authored comprehensive `docs/TUTORIAL.md`.
- [x] **Restore ML Components** — Verified ML curator logic and feature extraction.
- [x] **Refine Test Suite** — Resolved typed object attribute mismatches and resumable extraction bugs.

### 🟢 Priority 3: Continue Parity Work (PARITY-002/003)
> **PAUSED:** Parity-specific tasks are paused until the Priority 1 & 2 product health milestones are fully green and locked in. 

The remaining 11.3% gap is a **structural tie-breaking divergence**. When multiple voxels on the watershed expansion frontier have identical penalized energies, Python and MATLAB must select the exact same voxel. 

- [ ] **Hub vertex tie-breaking** — Add a secondary sort key (Fortran-order linear index) to the frontier priority queue in `global_watershed.py`.
- [ ] **Strel loop order verification** — Confirm Python's `(Z, X, Y)` scanline matches MATLAB's argmin behavior for energy ties.
- [ ] **Candidate filtering alignment (Measure 3)** — Tighten Python acceptance criteria to match MATLAB `get_edges_by_watershed` filters in `candidate_generation.py` and `cleanup.py`.
- [ ] **Bit-Accurate Precision** — Validate float64 math remains stable across all edge tracing stages.
- [ ] **Run full proof** — `prove-exact --stage all` once edges exceed 95%.

### ⚪ Priority 4: Future Work
- [ ] **Performance (PERF-001)** — Resume `O(N²) → O(log N)` frontier optimization after parity stabilizes.
- [ ] **Dataset expansion** — Retrieve full TIFF volumes via `git annex get external/` for multi-dataset stability testing.
- [ ] **Curation Alignments** — Verify ML feature extraction and automated thresholds are perfectly aligned.
- [ ] **Production Release** — Promote the stable vectorization engine to standard academic & research deployment with green CI/CD.