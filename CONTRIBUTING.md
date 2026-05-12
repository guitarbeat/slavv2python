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


## Making Changes

- Follow the repo's [AGENTS.md](AGENTS.md) for canonical workflows and test placement.
- Use:
  
   ```powershell
    python -m ruff check slavv_python tests --fix
    python -m ruff format slavv_python tests
   ```

   for linting/formatting.
- Add or update tests in `tests/` as appropriate.
- If your change affects the public workflow, keep the default `paper` profile,
  authoritative `network.json`, and related docs aligned.

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

- Check [README.md](README.md) and [docs/README.md](docs/README.md) for more info.
- Open an issue for questions or suggestions.
