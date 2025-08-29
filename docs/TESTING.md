# Testing

This guide explains how to run quick compilation checks and the test suite.

## Quick compile check

Verify all Python files compile:

```bash
python -m py_compile $(git ls-files '*.py')
```

## Running tests

The tests expect the source path `slavv-streamlit/src` to be on `PYTHONPATH` so modules like `vectorization_core` can be imported directly.

From the repo root:

```bash
export PYTHONPATH=slavv-streamlit/src
pytest -q
```

Or for a single test module:

```bash
export PYTHONPATH=slavv-streamlit/src
pytest -q tests/test_public_api.py
```

## Tips
- Start with specific tests close to recent changes, then broaden.
- Large volumes: prefer synthetic fixtures; avoid committing large binary test data.
- For performance-heavy tests, run locally and gate CI with lighter smoke tests.

