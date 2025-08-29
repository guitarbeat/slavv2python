# Contributing

Thanks for your interest in contributing! This document outlines conventions for contributors and automated agents working on this repository.

## Documentation
- Place supplementary Markdown files in the `docs/` directory.
- Use relative links when referencing these documents from elsewhere in the repo.

## Code Style
- Follow basic PEP 8 formatting for Python code.

## Programmatic Checks
- After modifying any Python files, ensure they compile successfully:
  ```bash
  python -m py_compile $(git ls-files '*.py')
  ```
- Run tests from the repo root (source path on `PYTHONPATH`):
  ```bash
  export PYTHONPATH=slavv-streamlit/src
  pytest -q
  ```

## Commit Messages
- Provide concise summaries of the changes made.

## Where to Start
- See `docs/index.md` for an overview of docs and `docs/MATLAB_TO_PYTHON_MAPPING.md` for porting status.
