# Folder Purpose Guide

**Why do we have four top-level folders?** This guide clarifies the distinct purposes of `slavv_python/`, `tests/`, `scripts/`, and `workspace/`.

## Quick Reference

| Folder | What Lives There | Who Uses It | Committed to Git |
|:-------|:-----------------|:------------|:-----------------|
| **`slavv_python/`** | Production package code | End users + developers | ✅ Yes |
| **`tests/`** | Automated test suite | Developers + CI | ✅ Yes |
| **`scripts/`** | Developer utility scripts | Developers only | ✅ Yes |
| **`workspace/`** | Experiment artifacts | Developer locally | ❌ No (.gitignore) |

---

## slavv_python/ — The Product

**Purpose:** The installable Python package that end users get when they run `pip install slavv`.

**Contains:**
- `engine/` — Pipeline orchestration and run lifecycle
- `processing/` — Scientific computation (energy, vertices, edges, network)
- `analytics/` — Analysis tools and parity proof harness
- `storage/` — Data I/O (TIFF loading, JSON export)
- `interface/` — User-facing surfaces (CLI, Streamlit app)
- `visualization/` — Plotting and rendering
- `workflows/` — Pipeline orchestration and profiles
- `schema/` — Data models
- `utils/` — Validation, math, formatting

**Key Trait:** This is **library code**. Every file here should be production-quality, tested, typed, and usable by external consumers.

**Entry Points:**
```python
from slavv_python import SlavvPipeline, load_tiff_volume
```
```powershell
slavv run -i volume.tif -o output
slavv-app
```

---

## tests/ — Verification

**Purpose:** Automated test suite that proves `slavv_python/` works correctly.

**Contains:**
- `unit/` — Fast, isolated tests by owning module
- `integration/` — Cross-component and end-to-end pipeline tests
- `integration/parity/` — MATLAB parity pre-gate tests
- `ui/` — Streamlit and visualization tests
- `support/` — Shared test builders and fixtures

**Key Trait:** These tests exercise the **production code** in `slavv_python/`. They don't contain any production logic themselves.

**Run Via:**
```powershell
python -m pytest tests/
python -m pytest -m "unit or integration"
```

---

## scripts/ — Developer Utilities

**Purpose:** One-off tools and experimental harnesses used **during development** but not part of the public product.

**Contains:**
- `cli/parity_experiment.py` — MATLAB oracle promotion and exact-proof runner
- `cli/compare_execution_traces.py` — Debug helper for trace analysis
- `diagnostics/` — MATLAB artifact inspection tools
- `matlab/` — Headless MATLAB drivers for oracle vectorization

**Key Trait:** These are **maintained utilities** for developers, not production code. They're committed to Git because they're reusable, but they're not installed when users `pip install slavv`.

**When to Add Here:**
- Parity experiment harnesses (e.g., `parity_experiment.py`)
- MATLAB oracle promotion workflows
- Diagnostic tools for investigating specific issues
- Benchmarking scripts

**When NOT to Add Here:**
- Production CLI commands (put in `slavv_python/interface/cli/`)
- Throwaway one-time exploration scripts (put in `workspace/scratch/`)
- Test fixtures (put in `tests/support/`)

---

## workspace/ — Experiment Artifacts

**Purpose:** Local developer experiment data that changes constantly and should **never** be committed to Git.

**Contains:**
- `oracles/` — Preserved MATLAB truth vectors for parity testing
- `runs/` — Trial pipeline runs with checkpoints and outputs
- `reports/` — Promoted proof summaries
- `datasets/` — Test datasets and sample volumes
- `scratch/` — Temporary files, logs, one-off scripts

**Key Trait:** This is your **personal lab notebook**. Every developer has their own `workspace/` with different contents. It's in `.gitignore`.

**When to Add Here:**
- Pipeline run outputs (`workspace/runs/my_experiment/`)
- MATLAB oracle vectors (`workspace/oracles/180709_crop_M/`)
- Parity proof reports (`workspace/reports/crop_M_exact_proof.md`)
- Temporary exploration scripts (`workspace/scratch/quick_check.py`)
- Intermediate datasets (`workspace/datasets/crop_for_testing.tif`)

**When NOT to Add Here:**
- Anything that other developers need (belongs in `slavv_python/`, `tests/`, or `scripts/`)
- Documentation (belongs in `docs/`)
- Reusable test fixtures (belongs in `tests/support/`)

---

## Decision Tree: Where Does My Code Go?

```
┌─────────────────────────────────────────┐
│ I just wrote some code. Where does it   │
│ go?                                      │
└─────────────┬───────────────────────────┘
              │
              ▼
     ┌────────────────────┐
     │ Is it part of the  │ YES ──▶ slavv_python/
     │ public product?    │
     └────────┬───────────┘
              │ NO
              ▼
     ┌────────────────────┐
     │ Does it test the   │ YES ──▶ tests/
     │ production code?   │
     └────────┬───────────┘
              │ NO
              ▼
     ┌────────────────────┐
     │ Is it a reusable   │ YES ──▶ scripts/
     │ developer tool?    │
     └────────┬───────────┘
              │ NO
              ▼
     ┌────────────────────┐
     │ It's a one-off     │
     │ experiment or      │ ──────▶ workspace/scratch/
     │ temporary file     │
     └────────────────────┘
```

---

## Examples

### Example 1: New Edge Detection Algorithm
**Location:** `slavv_python/processing/stages/edges/my_new_algorithm.py`  
**Why:** It's part of the production pipeline that end users can select.

### Example 2: Test for New Algorithm
**Location:** `tests/unit/core/test_my_new_algorithm.py`  
**Why:** It verifies production code behavior.

### Example 3: Script to Compare Two Algorithms
**Location:** `scripts/compare_edge_algorithms.py`  
**Why:** Reusable developer tool, not part of public product.

### Example 4: Quick One-Off Investigation
**Location:** `workspace/scratch/debug_edge_issue.py`  
**Why:** Temporary exploration, not intended for reuse.

### Example 5: Pipeline Run Output
**Location:** `workspace/runs/crop_M_exact/`  
**Why:** Experiment artifact, changes frequently, not committed.

### Example 6: MATLAB Oracle Vectors
**Location:** `workspace/oracles/180709_crop_M/`  
**Why:** Large binary artifacts for local parity testing.

---

## Common Confusion Points

### "Should parity_experiment.py be in slavv_python/?"
**No.** It's a developer-only tool for MATLAB parity work, not part of the public product. It stays in `scripts/` and is tested via `tests/unit/scripts/`.

### "Should I commit my workspace/ folder?"
**No.** It's in `.gitignore` because it contains large datasets, personal experiment runs, and temporary files that differ for every developer.

### "Can scripts/ import from slavv_python/?"
**Yes.** Scripts are allowed to import the production package. But `slavv_python/` should **never** import from `scripts/`.

### "Where do MATLAB parity tests go?"
- **Pre-gate integration tests:** `tests/integration/parity/`
- **Parity harness utilities:** `slavv_python/analytics/parity/` (production code)
- **Parity experiment runner:** `scripts/parity_experiment.py` (developer tool)
- **Oracle promotion:** `scripts/matlab/` (MATLAB driver) + `scripts/parity_experiment.py promote-oracle`

### "Where do one-off exploration scripts go?"
**`workspace/scratch/`** — These are throwaway scripts you write to investigate something specific. They're not committed and not maintained.

---

## Summary

| Folder | Purpose | Committed | Installed |
|:-------|:--------|:----------|:----------|
| `slavv_python/` | Production package | ✅ Yes | ✅ Yes |
| `tests/` | Test suite | ✅ Yes | ❌ No |
| `scripts/` | Developer tools | ✅ Yes | ❌ No |
| `workspace/` | Experiment data | ❌ No | ❌ No |

The key insight: **Every folder has a different lifecycle and audience.**

- End users only see `slavv_python/`.
- Developers use all four folders.
- Git only tracks `slavv_python/`, `tests/`, and `scripts/`.
- `workspace/` is your personal experiment space.
