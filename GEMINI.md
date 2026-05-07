# AI Agent Repository Guide (GEMINI.md)

This document provides canonical instructions, constraints, workflows, and guardrails for any AI coding agent working in the `slavv2python` repository. All agents must read and strictly adhere to these guidelines to ensure codebase consistency, exact mathematical/behavioral parity with MATLAB, and robust software architecture.

---

## 🎯 Scope & Core Principles

- **Work Location:** Always work from the repository root directory.
- **Environment:** Prefer Windows PowerShell-friendly commands.
- **Canonical Source of Truth:** Treat the workflows, commands, and rules in this file as the definitive guidance for the repo.

---

## 🗺️ Repository Map

| Path | Purpose / Description |
| :--- | :--- |
| `source/` | Core package code (processing, I/O, analysis, visualization, app entry points). |
| `dev/tests/` | Test suite covering unit, integration, UI, and diagnostics. |
| `dev/scripts/` | Maintained developer helper scripts, runners, and benchmarks. |
| `docs/` | Maintained reference documentation for the Python codebase. |

---

## 📖 Key Reference Documents

Always read these files first when working on relevant surfaces:

- **Index:** [docs/README.md](file:///d:/2P_Data/Aaron/slavv2python/docs/README.md) — Index for all maintained reference docs.
- **MATLAB Parity Plan:** [docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md](file:///d:/2P_Data/Aaron/slavv2python/docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md) — Canonical claim boundaries, source-of-truth hierarchy, and remaining implementation work.
- **MATLAB-to-Python Map:** [docs/reference/core/MATLAB_PARITY_MAPPING.md](file:///d:/2P_Data/Aaron/slavv2python/docs/reference/core/MATLAB_PARITY_MAPPING.md) — Function-to-function mapping for exact parity.
- **Python Naming Guide:** [docs/reference/workflow/PYTHON_NAMING_GUIDE.md](file:///d:/2P_Data/Aaron/slavv2python/docs/reference/workflow/PYTHON_NAMING_GUIDE.md) — Preferred Python naming conventions and package surfaces.
- **Testing Guide:** [dev/tests/README.md](file:///d:/2P_Data/Aaron/slavv2python/dev/tests/README.md) — Rules for test placement and markers.
- **Shared Test Config:** [dev/tests/conftest.py](file:///d:/2P_Data/Aaron/slavv2python/dev/tests/conftest.py) — Shared pytest behavior and `tmp_path` setup.
- **Extraction Algorithms:** [docs/reference/workflow/ADDING_EXTRACTION_ALGORITHMS.md](file:///d:/2P_Data/Aaron/slavv2python/docs/reference/workflow/ADDING_EXTRACTION_ALGORITHMS.md) — Contributor guide for adding new extraction algorithms.
- **Runtime Tracking:** `source/runtime/run_tracking/` — Structured run metadata and staged artifact locations.

---

## ⚙️ Setup & Installation

To set up or recreate the local environment:

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependency set matching your task
pip install -e .                # Core package only
pip install -e ".[app]"          # With Streamlit app dependencies
pip install -e ".[app,dev]"      # Full developer environment (recommended)

# Install pre-commit hooks
pre-commit install
```

---

## 🛠️ Canonical Quality Commands

### Code Formatting
```powershell
python -m ruff format source dev/tests
python -m ruff format --check source dev/tests
```

### Linting & Auto-fixes
```powershell
python -m ruff check source dev/tests
python -m ruff check source dev/tests --fix
```

### Type Checking
```powershell
python -m mypy
```
> [!NOTE]
> The current `mypy` gate is run from the repo-root and covers the CLI, Streamlit launcher, share-report, web app, run-state, and selected core pipeline modules.

### Running Tests
```powershell
python -m pytest dev/tests/
python -m pytest -m "unit or integration"
```

### Other Checks
```powershell
python -m compileall source dev/scripts
pre-commit run --all-files
```

---

## 🚀 CLI & Application Workflows

### Package CLI Commands
```powershell
slavv info
slavv run -i volume.tif -o slavv_output --export csv json
slavv run -i volume.tif -o slavv_output --profile matlab_compat --export json
slavv analyze -i slavv_output/network.json
slavv plot -i slavv_output/network.json -o plots.html
```

### Advanced `slavv run` Options
```powershell
slavv run -i volume.tif -o slavv_output --run-dir dev\runs\sample_a
slavv run -i volume.tif -o slavv_output --stop-after edges
slavv run -i volume.tif -o slavv_output --force-rerun-from vertices
```
*Note: `slavv run` writes structured run metadata under `<output>\_slavv_run` when `--run-dir` is omitted. The CLI defaults to the native `paper` profile.*

### Streamlit App
```powershell
slavv-app
python -m streamlit run source/apps/streamlit/app.py
```
*Note: The launcher requires the `app` extra. The ML curation flow accepts `.joblib` and `.pkl` model files directly.*

---

## 🔄 Recommended Workflows

### ⚡ Small Code Changes
1. Read the impacted module and its nearest tests first.
2. Verify test placement against `dev/tests/README.md` and `dev/tests/conftest.py`.
3. Run the smallest targeted `pytest` command covering the change.
4. Format and lint: `ruff check --fix` and `ruff format`.
5. Run full targeted suite: `python -m pytest -m "unit or integration"` if the change crosses module boundaries.

### 🛡️ Regression Checks
Run these checks before submitting substantial changes:
```powershell
python -m compileall source dev/scripts
python -m ruff format --check source dev/tests
python -m ruff check source dev/tests
python -m mypy
python -m pytest -m "unit or integration"
```
*Note: If the change is UI-facing, also run tests in `dev/tests/ui/`.*

### 🔬 Developer Parity Experiments
Use this flow to compare current code runs against a staged comparison root and a preserved MATLAB oracle package:
```powershell
# Promote oracle
python dev/scripts/cli/parity_experiment.py promote-oracle --matlab-batch-dir D:\incoming\batch_260421-151654 --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a --dataset-file D:\datasets\volume.tif --oracle-id v22_a

# Run preflight check
python dev/scripts/cli/parity_experiment.py preflight-exact --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\my_current_code_trial

# Prove lookup tables (LUTs)
python dev/scripts/cli/parity_experiment.py prove-luts --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\my_current_code_trial

# Run full exact proof comparison
python dev/scripts/cli/parity_experiment.py prove-exact --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\my_current_code_trial --stage all
```

---

## 🧮 Exact MATLAB Parity Rule

For any MATLAB-parity-sensitive surface (especially the `edges` and `network` stages), the required goal is **exact mathematical and algorithm-level parity**, not approximate behavioral similarity.

- **Truth Source:** Treat the MATLAB source under `external/Vectorization-Public/source/` as the canonical implementation.
- **Proof Gate:** Use `prove-exact` results and preserved MATLAB vectors as the proof gate.
- **No Approximations:** Do not accept "close enough" replacements (e.g., heuristic supplements, local tracing deviations) unless explicitly approved and documented.
- **1:1 Structure:** Python parity work must reproduce the same mathematical method and algorithm structure 1:1. Any undocumented deviation is a bug.

---

## 🚨 Repo-Specific Guardrails & Constraints

> [!IMPORTANT]
> **Max File Length:** Do not create or modify Python scripts to be more than **1000 lines** long. Keep files modular and focused.

- **Package Layout:** Keep all package code under `source/`. Use the grouped package surfaces described in `PYTHON_NAMING_GUIDE.md`.
- **Test Placement:** Keep tests under `dev/tests/` (following `dev/tests/README.md`). Files containing `regression` in the name automatically receive the `regression` marker.
- **Temporary Files:** Use the repo-local `tmp_path` fixture in `dev/tests/conftest.py` for testing. Keep test artifacts under `dev/tmp_tests/`, not ad-hoc system temp roots.
- **Logging:** Use the standard `logging` library in core/library code instead of `print()`. CLI commands may print user-facing console summaries.
- **Path Handling:** Prefer `pathlib.Path` for filesystem-heavy code and use explicit text encodings (e.g., `encoding="utf-8"`) when writing repository-managed text files.
- **Type Annotations:** Prefer `from __future__ import annotations` in all Python modules to match the prevailing package style.
- **CLI Framework:** Keep CLI surfaces aligned with the `argparse`-based entrypoints under `source/apps/`. Do not introduce new CLI frameworks.
- **Resumability:** Keep only the structured `run_dir` resumable surface; legacy checkpoint compatibility is not supported.
- **Search Exclusions:** When searching (e.g., with `rg`), exclude noisy generated directories like `dev/tmp_tests/` and vendored assets under `external/blender_resources/`.
