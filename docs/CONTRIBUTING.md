# Contributing to slavv2python

## In short

Fork, install, test, PR. If you change Energy/Vertices/Edges/Network, check
MATLAB match on the cheap tests first. Phase 1 already shipped — do not reopen
it for last-digit Energy diffs.

Setup, workflow, and PR guidelines for contributors.

## Setup

1. **Fork and clone the repository**
2. **Create a virtual environment and install dependencies**
  
   ```powershell
   uv sync --extra app
   pre-commit install
   ```
3. **Run tests before submitting a PR**

   ```powershell
   uv run pytest -m "unit or integration"
   ```


## Development Workflow

### Quality Gate
Before submitting a PR:
- **Linting**: `uv run ruff check slavv_python tests` passes (or auto-fixes).
- **Formatting**: `uv run ruff format slavv_python tests` is applied.
- **Type Checking**: `uv run mypy` is green.
- **Tests**: `uv run pytest` passes.

```powershell
pre-commit run --all-files
```

### Parity Testing
When modifying core vascular discovery logic (e.g. `matlab_get_edges_by_watershed.py`), verify MATLAB parity is maintained.

1.  **Preflight**: Prepare a parity experiment directory.
    ```powershell
    uv run slavv parity preflight-exact `
      --source-run-root workspace/runs/<last_known_good> `
      --oracle-root workspace/oracles/<dataset_id> `
      --dest-run-root workspace/runs/my_fix_trial
    ```
2.  **Prove**: Compare your changes against the oracle.
    ```powershell
    uv run slavv parity prove-exact `
      --source-run-root workspace/runs/my_fix_trial `
      --oracle-root workspace/oracles/<dataset_id> `
      --dest-run-root workspace/runs/my_fix_trial `
      --stage all
    ```
3.  Parity-sensitive changes must satisfy each stage's defined certification bar (ADR 0011 strict discrete + allclose floats for Energy/Vertices; ADR 0012 spatial bars for Edges/Network). Do not use strict-zero `prove-exact-sequence` failure as a Phase 1 reopen.

## Submitting a Pull Request

1. Create a branch, make your change, ensure tests and linting pass.
2. Open a PR against `main` with a clear description.
3. Link related issues if applicable.

## Code Review

- Address reviewer comments promptly.
- Ensure parity and regression tests pass if your change affects core or parity logic.
- Ensure CLI/app and export tests pass if your change affects the paper-facing run, analyze, plot, or Streamlit workflow.
