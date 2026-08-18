# Folder Purpose Guide

## In short

Code that users run lives in `slavv_python/`. Tests live in `tests/`. Maintained
docs live in `docs/`. Pictures for papers live in
`figures/`. The Experiment Root lives in `workspace/` (Oracles, datasets, live
Parity Runs). See [Experiment Root](../../../AGENTS.md#experiment-root). Scratch
under `workspace/scratch/` is local and gitignored.

**Why do we have separate top-level folders?** This guide clarifies the distinct purposes of `slavv_python/`, `tests/`, `docs/`, `figures/`, and `workspace/`.

The **repo root** stays a short front door: `README.md`, `AGENTS.md`, `STRATEGY.md`, `LICENSE`, `pyproject.toml`, and tool config (`.gitignore`, `.pre-commit-config.yaml`, …). Durable docs go in `docs/`. One-off reports, agent dumps, and lockfile experiments go in `workspace/scratch/`, not the root.

## Quick Reference

| Folder | What Lives There | Who Uses It | Committed to Git |
|:-------|:-----------------|:------------|:-----------------|
| **repo root** | Short front door: `README.md`, `AGENTS.md`, `STRATEGY.md`, `LICENSE`, `pyproject.toml`, tool config | Everyone | ✅ Yes |
| **`slavv_python/`** | Production package code | End users + developers | ✅ Yes |
| **`tests/`** | Automated test suite | Developers + CI | ✅ Yes |
| **`docs/`** | Maintained reference, ADRs, research notes | Developers + agents | ✅ Yes |
| **`figures/`** | Proposal / methods standalone figures + generators | Proposal / paper drafts | ✅ Yes |
| **`workspace/`** | Experiment Root: Oracles, datasets, live dests; scratch is local | Developers | USB/local binaries / ❌ scratch |

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

**Purpose:** Checked-in publication **standalone** claim figures (PDF/PNG) and
their generators for the PhD proposal appendix and related methods write-ups.

**Canonical inventory, captions, and regenerate commands:**
[figures/README.md](../../../figures/README.md).

**Contains (summary):**
- Claim charts: `parity_trajectory`, `parity_funnel`, `parity_agreement`, `parity_cert_table`
- Data + regenerator: `parity_campaign_series.py`, `generate_parity_claim_figures.py`
- Research drafts: [`figures/research/`](../../../figures/research/) (ULP / speedup)

**Not here:** runtime plotting (`slavv_python/visualization/`).

---

## scripts/ — Developer Utilities

> **Note:** Prefer CLI subcommands for product workflows:
> - Parity runner → `slavv parity <subcommand>`
> - Trace comparator → `slavv parity compare-traces`
> - Crop export → `slavv parity export-crop`
> - One-off diagnostics → `workspace/scratch/`
>
> Some checked-in generators remain under `scripts/` (e.g. `make_report_figures.py`)
> writing into `figures/research/`, and under `figures/` for claim figures.

---

## workspace/ — Experiment Root

See [Experiment Root](../../../AGENTS.md#experiment-root).

**Purpose:** On-disk home of Oracles, datasets, the live Claim Run Root, crop
guard, stretch dest, and archived proofs. GitHub does not carry the binaries
(not GitHub LFS). Copy by USB/rsync after clone. `scratch/` stays local.

**Contains:**
- `oracles/` — Preserved MATLAB truth vectors (local/USB)
- `runs/` — Live dests `canonical_full_v18`, `crop_M_exact_v3`, `crop_M_stretch_engine_v2` (local/USB); new writers stay local until promoted
- `reports/` — Archived proof JSON
- `datasets/` — Test volumes (local/USB)
- `scratch/` — Temporary files, logs, one-off scripts (gitignored)

**Key Trait:** Completeness is `slavv parity inspect-experiment-root` on this machine, not a GitHub LFS pull.

**When to Add Here:**
- Promoted Oracles and dataset TIFFs
- The live Claim Run Root / crop guard / stretch dest
- Temporary exploration scripts (`workspace/scratch/quick_check.py`)

**When NOT to Add Here:**
- Production code (`slavv_python/`)
- Documentation (`docs/`)
- Reusable test fixtures (`tests/support/`)
- Dual-write `checkpoint_edge_candidates.pkl` (read fallback only; gitignored)

Check completeness: `slavv parity inspect-experiment-root`.

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
**Location:** `workspace/runs/oracle_180709_E/crop_M_exact_v3/`  
**Why:** Live crop-guard dest (Experiment Root). New writers stay local until promoted.

### Example 6: MATLAB Oracle Vectors
**Location:** `workspace/oracles/180709_E_full_v2/`  
**Why:** Oracle binaries for `prove-exact`. Local/USB; see [Experiment Root](../../../AGENTS.md#experiment-root).

---

## Common Confusion Points

### "Should parity_experiment.py be in slavv_python/?"
**It's already there.** The parity tooling lives in `slavv_python/analytics/parity/` and is invoked via `slavv parity <subcommand>`. Developer probe scripts still live under `scripts/` (not the public CLI).

### "Should I commit my workspace/ folder?"
**Binaries, no. Scratch, no.** See [Experiment Root](../../../AGENTS.md#experiment-root). GitHub does not carry Oracle/dest/dataset binaries (not GitHub LFS). `workspace/scratch/` stays gitignored. After clone, copy binaries by USB/rsync and run `slavv parity inspect-experiment-root`.

### "Can scripts/ import from slavv_python/?"
**Yes.** `scripts/` contains developer probe scripts (e.g. `watershed_frontier_diff.py`, `edge_selection_funnel_probe.py`) referenced from [HANDOFF](../../../.claude/HANDOFF.md). They import from `slavv_python/` and are not part of the public CLI — use `slavv parity <subcommand>` for product workflows.

### "Where do MATLAB parity tests go?"
- **Pre-gate integration tests:** `tests/integration/parity/`
- **Parity harness utilities:** `slavv_python/analytics/parity/` (production code)
- **Parity experiment runner:** `slavv parity <subcommand>` (backed by `slavv_python/analytics/parity/`)
- **Oracle promotion:** `workspace/scratch/matlab/` (MATLAB driver) + `slavv parity promote-oracle`

### "Where do one-off exploration scripts go?"
**`workspace/scratch/`** — throwaway scripts, not committed, not maintained.
