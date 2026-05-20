# AI Agent Repository Guide

Canonical instructions for any AI coding agent working in the `slavv2python` repository.

---

## Scope & Core Principles

- **Work Location:** Always work from the repository root directory.
- **Environment:** Prefer Windows PowerShell-friendly commands.
- **Source of Truth:** This file is the definitive guidance. When in doubt, defer here.

---

## Repository Map

```
slavv2python/
├── slavv_python/                       # Main package
│   ├── engine/                         # Pipeline orchestration & lifecycle
│   │   └── state/                      # Run tracking, snapshots, resume
│   ├── processing/                     # Scientific computation
│   │   ├── image/                      # Normalization, tiling
│   │   └── stages/                     # Pipeline stages
│   │       ├── energy/                 # Hessian filtering, backends
│   │       ├── vertices/               # Extraction, painting, selection
│   │       ├── edges/                  # Watershed, tracing, selection, cleanup
│   │       │   └── matlab_algorithms/  # MATLAB-shaped parity shims
│   │       └── network/               # Strand assembly, graph construction
│   ├── analytics/                      # Analysis & metrics
│   │   ├── parity/                     # MATLAB exact proof harness
│   │   ├── curation/                   # Automated & ML curators
│   │   └── metrics/                    # Intensity, topology metrics
│   ├── storage/                        # Data I/O
│   │   ├── loaders/                    # TIFF, network loaders
│   │   └── exporters/                  # JSON v1 exporter
│   ├── interface/                      # User-facing surfaces
│   │   ├── cli/                        # argparse CLI
│   │   ├── streamlit/                  # Streamlit web app
│   │   └── shared_services/            # Cross-UI service layer
│   ├── visualization/                  # Plotting & rendering
│   ├── workflows/                      # Pipeline orchestration helpers, profiles
│   ├── schema/                         # Data models
│   └── utils/                          # Validation, math, formatting
│
├── tests/                              # Test suite
│   ├── unit/                           # By-owner unit tests
│   ├── integration/                    # End-to-end & parity tests
│   ├── ui/                             # Streamlit & visualization tests
│   ├── runtime/                        # Run-state management tests
│   └── support/                        # Shared test builders & fixtures
│
├── scripts/                            # Developer scripts
│   ├── cli/                            # Parity experiment harness
│   └── diagnostics/                    # MATLAB artifact inspection
│
├── docs/                               # Documentation
│   ├── reference/                      # Maintained technical references
│   └── investigations/                 # Archival investigation narratives
│
├── workspace/                          # Developer experiment workspace
│   ├── oracles/                        # Preserved MATLAB oracle vectors
│   ├── runs/                           # Experiment trial runs
│   ├── reports/                        # Promoted proof summaries
│   ├── datasets/                       # Test datasets
│   └── scratch/                        # Temporary scratch files
│
└── external/                           # Vendored dependencies
    └── Vectorization-Public/           # Canonical MATLAB source (submodule)
```

---

## Key Reference Documents

Read these first when working on relevant surfaces:

| Document | Path | Purpose |
|:---------|:-----|:--------|
| Doc Index | [docs/README.md](docs/README.md) | Index for all maintained reference docs |
| MATLAB Parity Plan | [docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md](docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md) | Claim boundaries, source-of-truth hierarchy, remaining work |
| MATLAB-to-Python Map | [docs/reference/core/MATLAB_PARITY_MAPPING.md](docs/reference/core/MATLAB_PARITY_MAPPING.md) | Function-to-function mapping for exact parity |
| Naming Guide | [docs/reference/workflow/PYTHON_NAMING_GUIDE.md](docs/reference/workflow/PYTHON_NAMING_GUIDE.md) | Python naming conventions and package surfaces |
| Testing Guide | [tests/README.md](tests/README.md) | Rules for test placement and markers |
| Extraction Algorithms | [docs/reference/workflow/ADDING_EXTRACTION_ALGORITHMS.md](docs/reference/workflow/ADDING_EXTRACTION_ALGORITHMS.md) | Contributor guide for new algorithms |
| Project Status | [PROJECT_STATUS.md](PROJECT_STATUS.md) | Comprehensive codebase health & parity status |

---

## Setup & Installation

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependency set matching your task
pip install -e .                       # Core package only
pip install -e ".[app]"                # With Streamlit app dependencies
pip install -e ".[app,workspace]"      # Full developer environment (recommended)

# Install pre-commit hooks
pre-commit install
```

---

## Quality Commands

### Formatting & Linting
```powershell
python -m ruff format slavv_python tests
python -m ruff check slavv_python tests --fix
```

### Type Checking
```powershell
python -m mypy
```

### Running Tests
```powershell
python -m pytest tests/
python -m pytest -m "unit or integration"
```

### Full Regression Gate
Run before substantial changes:
```powershell
python -m compileall slavv_python scripts
python -m ruff format --check slavv_python tests
python -m ruff check slavv_python tests
python -m mypy
python -m pytest -m "unit or integration"
```

---

## CLI & Application Workflows

### Core CLI
```powershell
slavv info
slavv run -i volume.tif -o slavv_output --export csv json
slavv run -i volume.tif -o slavv_output --profile matlab_compat --export json
slavv analyze -i slavv_output/network.json
slavv plot -i slavv_output/network.json -o plots.html
```

### Advanced Options
```powershell
slavv run -i volume.tif -o slavv_output --run-dir workspace\runs\sample_a
slavv run -i volume.tif -o slavv_output --stop-after edges
slavv run -i volume.tif -o slavv_output --force-rerun-from vertices
```

> [!NOTE]
> `slavv run` writes structured run metadata under `<output>\_slavv_run` when `--run-dir` is omitted. The CLI defaults to the native `paper` profile.

### Streamlit App
```powershell
slavv-app
python -m streamlit run slavv_python/interface/streamlit/app.py
```

---

## Developer Workflows

### Small Code Changes
1. Read the impacted module and its nearest tests.
2. Verify test placement against `tests/README.md`.
3. Run the smallest targeted `pytest` command.
4. Format and lint: `ruff check --fix` and `ruff format`.
5. Run full suite if the change crosses module boundaries.

### Parity Experiments
```powershell
# Promote oracle
python scripts/cli/parity_experiment.py promote-oracle \
  --matlab-batch-dir D:\incoming\batch_260421-151654 \
  --oracle-root workspace\oracles\v22_a \
  --dataset-file D:\datasets\volume.tif \
  --oracle-id v22_a

# Run preflight check
python scripts/cli/parity_experiment.py preflight-exact \
  --source-run-root workspace\runs\seed_run \
  --oracle-root workspace\oracles\v22_a \
  --dest-run-root workspace\runs\my_current_code_trial

# Run full exact proof comparison
python scripts/cli/parity_experiment.py prove-exact \
  --source-run-root workspace\runs\seed_run \
  --oracle-root workspace\oracles\v22_a \
  --dest-run-root workspace\runs\my_current_code_trial \
  --stage all
```

---

## Exact MATLAB Parity Rule

For any MATLAB-parity-sensitive surface (especially `edges` and `network` stages):

- **Truth Source:** The MATLAB code under `external/Vectorization-Public/` is the canonical implementation.
- **Proof Gate:** Use `prove-exact` results and preserved MATLAB oracle vectors.
- **No Approximations:** Do not accept "close enough" replacements unless explicitly approved and documented.
- **1:1 Structure:** Python parity work must reproduce the same mathematical method and algorithm structure. Any undocumented deviation is a bug.

---

## Guardrails & Constraints

> [!IMPORTANT]
> **Max File Length:** Do not create or modify Python scripts to be more than **1000 lines** long.

| Rule | Detail |
|:-----|:-------|
| **Package Layout** | All package code under `slavv_python/`. Use surfaces from `PYTHON_NAMING_GUIDE.md`. |
| **Test Placement** | Tests under `tests/` per `tests/README.md`. Files with `regression` in the name get the `regression` marker automatically. |
| **Temporary Files** | Use the repo-local `tmp_path` fixture from `tests/conftest.py`. Test artifacts go in `tmp_tests/`. |
| **Logging** | Use `logging` in library code, not `print()`. CLI may print user-facing summaries. |
| **Path Handling** | Prefer `pathlib.Path`. Use explicit `encoding="utf-8"` for text files. |
| **Type Annotations** | Prefer `from __future__ import annotations` in all modules. |
| **CLI Framework** | Keep CLI under `slavv_python/interface/cli/` (argparse). No new CLI frameworks. |
| **Resumability** | Only the structured `run_dir` surface; no legacy checkpoint compatibility. |
| **Search Exclusions** | Exclude `tmp_tests/`, `external/blender_resources/`, and cache directories when searching. |
| **Scratch Files** | One-off scripts, logs, and experiment artifacts go in `workspace/scratch/`, not the repo root. |
