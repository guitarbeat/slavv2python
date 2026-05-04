# Developer Guide for slavv2python

This document provides project-specific instructions for advanced developers working on the `slavv2python` repository.

## Build and Configuration

The project uses `setuptools` with a `pyproject.toml` configuration. It supports several optional dependency groups (extras).

### Environment Setup

1. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies based on your needs:**
   - Core only: `pip install -e .`
   - With Streamlit app: `pip install -e ".[app]"`
   - Full development suite: `pip install -e ".[app,dev]"`
   - All extras (ML, notebooks, DICOM, etc.): `pip install -e ".[all]"`

3. **Install pre-commit hooks:**
   ```powershell
   pre-commit install
   ```

## Testing Information

### Configuration

Tests are managed by `pytest`. Configuration is located in `pytest.ini` and `dev/tests/conftest.py`. The project uses a custom `tmp_path` fixture that roots temporary test artifacts under `dev/tmp_tests/`.

### Running Tests

- **Run all unit and integration tests:**
  ```powershell
  python -m pytest -m "unit or integration"
  ```
- **Run tests for a specific module:**
  ```powershell
  python -m pytest dev/tests/unit/core/
  ```

### Adding New Tests

- **Placement:** Follow the mapping in `dev/tests/README.md`. Tests should mirror the `source/` package structure under `dev/tests/unit/`, `dev/tests/integration/`, etc.
- **Markers:** Use markers defined in `conftest.py`. Common markers include `unit`, `integration`, `regression`, and `ui`.
- **Fixtures:** Leverage shared fixtures in `dev/tests/fixtures/` and `dev/tests/conftest.py`.

### Demonstration Test

A simple test to verify the environment can be found at `dev/tests/unit/core/test_simple_demo.py`:

```python
import pytest
import numpy as np
from source.core.pipeline import SlavvPipeline

@pytest.mark.unit
def test_pipeline_initialization():
    pipeline = SlavvPipeline()
    assert pipeline is not None
    assert hasattr(pipeline, 'run')
```

Execute it using:
```powershell
python -m pytest dev/tests/unit/core/test_simple_demo.py
```

## Additional Development Information

### Code Style and Quality

The project enforces strict linting and formatting via `ruff` and type checking via `mypy`.

- **Format code:**
  ```powershell
  python -m ruff format source dev/tests
  ```
- **Lint code:**
  ```powershell
  python -m ruff check source dev/tests --fix
  ```
- **Type check:**
  ```powershell
  python -m mypy
  ```
  Note: `mypy` configuration in `pyproject.toml` targets specific entry points and core modules for gradual typing.

### MATLAB Parity

For modules involving imported MATLAB logic (especially in `edges` and `network` stages), maintaining 1:1 mathematical parity with the original MATLAB source (`external/Vectorization-Public/source/`) is mandatory. Refer to `docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md` for claim boundaries.

### Logging

Use the standard `logging` module in library code (`source/`). Avoid `print()` statements except for CLI-facing summaries.
