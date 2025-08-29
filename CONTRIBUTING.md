# Contributing

Thanks for your interest in contributing! This document outlines conventions for contributors and automated agents working on this repository.

## Documentation
- Place supplementary Markdown files in the `docs/` directory.
- Use relative links when referencing these documents from elsewhere in the repo.

## Code Style
- Follow basic PEP 8 formatting for Python code.

## Programmatic Checks & Testing
- Compile check for all Python files:
  ```bash
  python -m py_compile $(git ls-files '*.py')
  ```
- Run tests from the repo root (ensure source path on `PYTHONPATH`):
  ```bash
  export PYTHONPATH=slavv-streamlit/src
  pytest -q
  ```
  To run one test module:
  ```bash
  export PYTHONPATH=slavv-streamlit/src
  pytest -q tests/test_public_api.py
  ```

## Commit Messages
- Provide concise summaries of the changes made.

## Where to Start
- See `docs/MATLAB_TO_PYTHON_MAPPING.md` for canonical porting status (includes coverage and deviations).
