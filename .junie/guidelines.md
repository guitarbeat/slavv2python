# SLAVV Development Guidelines (Junie)

This document provides project-specific guidelines for Junie (AI Agent) working on the `slavv2python` repository.

## Core Context & Navigation

Before starting any task, review these sources in order:
1.  **[README.md](README.md)**: Project overview and quick start.
2.  **[AGENTS.md](AGENTS.md)**: **Canonical workflow commands** and repository-wide guardrails.
3.  **[docs/README.md](docs/README.md)**: Index for maintained reference docs.
4.  **[docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md](docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md)**: Canonical source-of-truth hierarchy, claim boundaries, and remaining work for fully implementing the released SLAVV method in Python.
5.  **[docs/reference/core/MATLAB_PARITY_MAPPING.md](docs/reference/core/MATLAB_PARITY_MAPPING.md)**: Stage-by-stage MATLAB-to-Python map for the imported-MATLAB exact route.

## Build & Configuration

### Environment Setup
1.  **Virtual Environment**: `python -m venv .venv` and `.\.venv\Scripts\Activate.ps1`.
2.  **Installation**: `pip install -e ".[app,dev]"` (includes all extras).
3.  **Pre-commit**: `pre-commit install`.

### Common Commands
Always prefer `python -m` for tool execution:
- **Format**: `python -m ruff format source dev/tests`
- **Lint**: `python -m ruff check source dev/tests --fix`
- **Type-check**: `python -m mypy`
- **Tests**: `python -m pytest dev/tests/`

## Repo-Specific Guardrails

- **Package Root**: Keep package code under `source/slavv/`.
- **Test Placement**: Follow `dev/tests/README.md`; mirror the `source/slavv/` structure.
- **Logging**: Use the standard `logging` module in library code; avoid `print()`.
- **Path Handling**: Prefer `pathlib.Path` and explicit `encoding="utf-8"`.
- **Code Style**: Use `from __future__ import annotations` in new modules.
- **CLI**: Align with `argparse`-based entrypoints in `source/slavv/apps/`.
- **Imports**: Preserve the `source/` package layout and existing console entrypoints (`slavv`, `slavv-app`).

## MATLAB Parity & Comparison

When working on parity-sensitive logic:
- **Canonical Source Rule**: Treat `external/Vectorization-Public/source/` as the source of truth.
- **Hierarchy Rule**: Use the released MATLAB source first, preserved MATLAB artifacts plus `prove-exact` second, and paper prose as explanatory context.
- **Exact Method Rule**: The goal is exact mathematical and algorithmic parity with MATLAB, not
  approximate output similarity.
- **Deviation Policy**: Heuristic replacements, post-hoc supplements, salvage passes, reordered
  claim resolution, or simplified local workflows are bugs on parity surfaces unless they are
  explicitly documented as non-parity behavior.
- **Staged Layout**: Respect the `01_Input/`, `02_Output/`, `03_Analysis/`, `99_Metadata/` structure.
- **Developer Runner**: Use `dev/scripts/cli/parity_experiment.py` for Python reruns and exact artifact proof against reusable staged comparison roots.
- **Evidence**: Cite artifacts from the staged comparison root you are using and from `docs/reference/core/MATLAB_PARITY_MAPPING.md` when describing source mappings.
- **Claim Boundary**: Do not call a stage `Exact` or `100%` until it is artifact-proven under the current maintained proof workflow.
- **Scope**: Rich legacy parity diagnostics are retired from the live source tree; the maintained runner supports rerun summaries plus exact artifact proof, but not the removed legacy diagnostics.

## Testing Strategy

- **Markers**: Use folder-based markers (`unit`, `integration`, `diagnostic`) as defined in `dev/tests/conftest.py`.
- **Regression**: Add targeted tests for every bug fix (e.g., `dev/tests/unit/core/test_watershed_supplement_regression.py`).
- **Temporary Files**: Use the repo-local `tmp_path` fixture which roots under `dev/tmp_tests/`.

### Setup Verification
To verify your environment, run the diagnostic suite:
```powershell
python -m pytest dev/tests/diagnostic/test_comparison_setup.py
```

