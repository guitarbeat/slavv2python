# SLAVV Python Roadmap & Status

**Date:** 2026-05-28  
**Version:** 0.1.0 (Beta)  
**Python:** ≥3.11  
**License:** GPL-3.0

Narrative roadmap and project status for **SLAVV (Strand Localization and Vessel Vectorization)**. **Open tasks and checkboxes live only in [TODO.md](TODO.md)** — do not add new task lists here.

## Navigation

| Link | Purpose |
| :--- | :--- |
| **[Developer dashboard (tasks)](TODO.md)** | Active checkboxes, Phase 1 status, planning hub |
| [Live proof status](reference/core/EXACT_PROOF_FINDINGS.md) | Per-stage exact-parity state and blockers |
| [Phase 1 spec](plans/phase-1-exact-route-spec.md) | Requirements + implementation for exact-route certification |
| [Investigation findings](investigations/translation_pair_analysis/INVESTIGATION_FINDINGS.md) | Historical missing/extra pair analysis |
| [Policy & implementation phases](reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md) | Claim boundaries and long-term phases |
| [Parity mapping](reference/core/MATLAB_PARITY_MAPPING.md) | Module-for-module structure against MATLAB |

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
| 3 | **Edges** | Global watershed → tracing → selection | ✅ Complete | 🟡 In progress | v29 baseline ~88.7% pair match; Phase 1 bar is **strict zero** via `prove-exact` |
| 4 | **Network** | Graph assembly & strand smoothing | ✅ Complete | ⏳ Pending | End-to-end verified; sequential proof pending upstream gates |

**Public workflow verified (2026-05-21):** `slavv run`, `slavv analyze`, `slavv plot`, and `slavv-app` (Streamlit).

**Current parity focus:** Phase 1 exact-route [Certification](AGENTS.md#certification) on full `180709_E` — **status:** [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md) · **tasks:** [TODO.md](TODO.md). Informal edge match rate is a diagnostic baseline, not the exit criterion.

## Strategy (unchanged)

The parity track moved from ~14% to ~88.7% pair match on edges, while the public product path stays stable. **Product-first, parity-parallel:** paper workflow remains shippable; exact route pursues MATLAB-equivalence under strict gates.

## Completed milestones (archive)

| Era | ID | Outcome |
|-----|-----|---------|
| 2026-05-21 | PAPER-001 | End-to-end `slavv run` / analyze / plot / app; paper-profile CI integration test |
| 2026-05-21 | Stabilization | Quality gate green (ruff, mypy, pytest); typed results migration |
| 2026-05-22 | PARITY-002/003 | Bit-accurate tie-breaking, strel alignment, distance cutoffs, precision fixes |

Architectural alignment on edges is **done**; remaining work is **certification** (sequential `prove-exact-sequence`), tracked in [TODO.md](TODO.md).

## Deferred themes (not scheduled here)

Performance (`O(log N)` frontier), neurovasc-db dataset expansion, ML curation alignment, and production release are listed in [TODO.md](TODO.md) under “next” when Phase 1 gates close.