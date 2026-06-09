# Contributing to slavv2python

Setup, workflow, and PR guidelines for contributors.

## Setup

1. **Fork and clone the repository**
2. **Create a virtual environment and install dependencies**
  
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -e ".[app,workspace]"
   pre-commit install
   ```
3. **Run tests before submitting a PR**

   ```powershell
   python -m pytest -m "unit or integration"
   ```


## Development Workflow

### Quality Gate
Before submitting a PR:
- **Linting**: `python -m ruff check slavv_python tests` passes (or auto-fixes).
- **Formatting**: `python -m ruff format slavv_python tests` is applied.
- **Type Checking**: `python -m mypy` is green.
- **Tests**: `python -m pytest` passes.

```powershell
pre-commit run --all-files
```

### Parity Testing
When modifying core vascular discovery logic (e.g. `global_watershed.py`), verify MATLAB parity is maintained.

1.  **Preflight**: Prepare a parity experiment directory.
    ```powershell
    slavv parity preflight-exact `
      --source-run-root workspace/runs/<last_known_good> `
      --oracle-root workspace/oracles/<dataset_id> `
      --dest-run-root workspace/runs/my_fix_trial
    ```
2.  **Prove**: Compare your changes against the oracle.
    ```powershell
    slavv parity prove-exact `
      --source-run-root workspace/runs/my_fix_trial `
      --oracle-root workspace/oracles/<dataset_id> `
      --dest-run-root workspace/runs/my_fix_trial `
      --stage all
    ```
3.  Parity-sensitive changes must preserve the strict-zero `prove-exact-sequence` bar. Historical match rates are diagnostics only.

## Submitting a Pull Request

1. Create a branch, make your change, ensure tests and linting pass.
2. Open a PR against `main` with a clear description.
3. Link related issues if applicable.

## Code Review

- Address reviewer comments promptly.
- Ensure parity and regression tests pass if your change affects core or parity logic.
- Ensure CLI/app and export tests pass if your change affects the paper-facing run, analyze, plot, or Streamlit workflow.
