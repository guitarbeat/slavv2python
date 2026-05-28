# Contributing to slavv2python

Thank you for your interest in contributing! This guide will help you get started, set up your environment, and submit changes effectively.

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

### 1. Quality Gate
We maintain a strict quality gate for the `main` branch. Before submitting a PR, ensure that:
- **Linting**: `python -m ruff check slavv_python tests` passes (or auto-fixes).
- **Formatting**: `python -m ruff format slavv_python tests` is applied.
- **Type Checking**: `python -m mypy` is green.
- **Tests**: `python -m pytest` passes with 100% success rate.

We recommend using `pre-commit` to automate these checks:
```powershell
pre-commit run --all-files
```

### 2. Parity Testing (Phase 3)
If you are modifying core vascular discovery logic (e.g., in `global_watershed.py`), you **must** verify that mathematical parity with MATLAB is maintained.

1.  **Run Preflight**: Prepare a parity experiment directory.
    ```powershell
    python scripts/cli/parity_experiment.py preflight-exact `
      --source-run-root workspace/runs/<last_known_good> `
      --oracle-root workspace/oracles/<dataset_id> `
      --dest-run-root workspace/runs/my_fix_trial
    ```
2.  **Execute Proof**: Compare your changes against the oracle.
    ```powershell
    python scripts/cli/parity_experiment.py prove-exact `
      --source-run-root workspace/runs/my_fix_trial `
      --oracle-root workspace/oracles/<dataset_id> `
      --dest-run-root workspace/runs/my_fix_trial `
      --stage all
    ```
3.  **Verify Match Rate**: Parity must remain **>95%** for core discovery changes to be accepted.

## Submitting a Pull Request

1. Create a new branch for your change.
2. Ensure all tests pass and code is linted.
3. Push your branch and open a Pull Request (PR) against `main`.
4. Fill out the PR template and describe your changes clearly.
5. Link related issues if applicable.

## Code Review

- Address reviewer comments promptly.
- Ensure parity and regression tests pass if your change affects core or parity logic.
- Ensure CLI/app and export tests pass if your change affects the paper-facing
  run, analyze, plot, or Streamlit workflow.

## Need Help?

- Check [README.md](../README.md) and [Documentation index](README.md) for more info.
- Open an issue for questions or suggestions.
