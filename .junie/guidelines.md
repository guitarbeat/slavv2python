# SLAVV Development Guidelines (Junie)

This document provides project-specific guidelines for Junie (AI Agent) working on the `slavv2python` repository.

## Core Context & Navigation

Before starting any task, review these sources in order:
1.  **[README.md](README.md)**: Project overview and quick start.
2.  **[AGENTS.md](AGENTS.md)**: **Canonical workflow commands** and repository-wide guardrails.
3.  **[docs/README.md](docs/README.md)**: Index for reference docs and **Active Chapter** status.
4.  **[TODO.md](TODO.md)**: Current technical blockers and canonical run roots for parity work.

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
- **Staged Layout**: Respect the `01_Input/`, `02_Output/`, `03_Analysis/`, `99_Metadata/` structure.
- **Comparison Tool**: Use `dev/scripts/cli/compare_matlab_python.py`.
- **Evidence**: Cite artifacts from canonical run roots defined in `TODO.md`.
- **Validation**: Use the `slavv parity-proof` command to verify exact parity.

## Testing Strategy

- **Markers**: Use folder-based markers (`unit`, `integration`, `diagnostic`) as defined in `dev/tests/conftest.py`.
- **Regression**: Add targeted tests for every bug fix (e.g., `dev/tests/unit/core/test_watershed_supplement_regression.py`).
- **Temporary Files**: Use the repo-local `tmp_path` fixture which roots under `dev/tmp_tests/`.

### Setup Verification
To verify your environment, run the diagnostic suite:
```powershell
python -m pytest dev/tests/diagnostic/test_comparison_setup.py
```

