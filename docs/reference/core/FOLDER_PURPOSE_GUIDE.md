# Folder Purpose Guide

**Why do we have separate top-level folders?** This guide clarifies the distinct purposes of `slavv_python/`, `tests/`, `docs/`, `figures/`, and `workspace/`.

## Quick Reference

| Folder | What Lives There | Who Uses It | Committed to Git |
|:-------|:-----------------|:------------|:-----------------|
| **`slavv_python/`** | Production package code | End users + developers | ✅ Yes |
| **`tests/`** | Automated test suite | Developers + CI | ✅ Yes |
| **`docs/`** | Maintained reference, ADRs, research notes | Developers + agents | ✅ Yes |
| **`figures/`** | Proposal / methods multipanel figures + generators | Proposal / paper drafts | ✅ Yes |
| **`workspace/`** | Experiment artifacts | Developer locally | ❌ No (.gitignore) |

---

## slavv_python/ — The Product

**Purpose:** The installable Python package that end users get when they run `pip install slavv`.

**Contains:**
- `engine/` — Pipeline orchestration and run lifecycle
- `pipeline/` — Scientific computation (energy, vertices, edges, network)
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

## figures/ — Proposal / methods figures

**Purpose:** Checked-in publication multipanels (PDF/PNG) and their generators for
the PhD proposal appendix and related methods write-ups.

**Contains (current):**
- `parity_trajectory.{pdf,png}` — crop candidate-pair overlap trajectory
- `parity_funnel.{pdf,png}` — crop edge-pair recovery funnel
- `parity_agreement.{pdf,png}` — canonical per-stage agreement
- `parity_cert_table.{pdf,png}` — ADR 0011/0012 gate status table
- `generate_matlab_python_parity_journey.py` — regenerator (four standalone figures)
- [README.md](../../../figures/README.md) — captions, evidence sources, regenerate command

**Not here:** runtime plotting (`slavv_python/visualization/`) or energy ULP/speedup
drafts from live run artifacts ([docs/research/figures/](../../research/figures/)).

---

## scripts/ — Developer Utilities

> **Note:** Prefer CLI subcommands for product workflows:
> - Parity runner → `slavv parity <subcommand>`
> - Trace comparator → `slavv parity compare-traces`
> - Crop export → `slavv parity export-crop`
> - One-off diagnostics → `workspace/scratch/`
>
> Some checked-in generators remain under `scripts/` (e.g. `make_report_figures.py`)
> and under `figures/` for proposal multipanels.

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
- Anything that other developers need (belongs in `slavv_python/` or `tests/`)
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
     │ It's a one-off     │
     │ experiment or      │ ──────▶ workspace/scratch/
     │ temporary file     │
     └────────────────────┘
```

---

## Examples

### Example 1: New Edge Detection Algorithm
**Location:** `slavv_python/pipeline/edges/my_new_algorithm.py`  
**Why:** Part of the production pipeline that end users can select.

### Example 2: Test for New Algorithm
**Location:** `tests/unit/pipeline/edges/test_my_new_algorithm.py`  
**Why:** Verifies production code behavior.

### Example 3: Script to Compare Two Algorithms
**Location:** `workspace/scratch/compare_edge_algorithms.py`  
**Why:** One-off comparison; not maintained or reusable.

### Example 4: Quick One-Off Investigation
**Location:** `workspace/scratch/debug_edge_issue.py`  
**Why:** Temporary exploration.

### Example 5: Pipeline Run Output
**Location:** `workspace/runs/crop_M_exact/`  
**Why:** Experiment artifact, changes frequently, not committed.

### Example 6: MATLAB Oracle Vectors
**Location:** `workspace/oracles/180709_crop_M/`  
**Why:** Large binary artifacts for local parity testing.

---

## Common Confusion Points

### "Should parity_experiment.py be in slavv_python/?"
**It's already there.** The parity tooling lives in `slavv_python/analytics/parity/` and is invoked via `slavv parity <subcommand>`. There is no longer a separate `scripts/` directory.

### "Should I commit my workspace/ folder?"
**No.** It's in `.gitignore` because it contains large datasets, personal experiment runs, and temporary files that differ for every developer.

### "Can scripts/ import from slavv_python/?"
**Not applicable** — `scripts/` no longer exists. All tooling is in `slavv_python/analytics/parity/` and invoked via `slavv parity <subcommand>`.

### "Where do MATLAB parity tests go?"
- **Pre-gate integration tests:** `tests/integration/parity/`
- **Parity harness utilities:** `slavv_python/analytics/parity/` (production code)
- **Parity experiment runner:** `slavv parity <subcommand>` (backed by `slavv_python/analytics/parity/`)
- **Oracle promotion:** `workspace/scratch/matlab/` (MATLAB driver) + `slavv parity promote-oracle`

### "Where do one-off exploration scripts go?"
**`workspace/scratch/`** — throwaway scripts, not committed, not maintained.
