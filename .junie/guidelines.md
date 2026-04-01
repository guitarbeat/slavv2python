# SLAVV Development Guidelines

This document provides project-specific guidelines for advanced developers working on the `slavv2python` repository.

## Build/Configuration Instructions

### Environment Setup
1. **Virtual Environment**: Create and activate a Python virtual environment (Python 3.9+ recommended).
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. **Installation**: Install the package with the necessary extras. For development and UI support:
   ```powershell
   pip install -e ".[app,dev]"
   ```
   Extras include:
   - `[app]`: Streamlit-based web interface (`slavv-app`).
   - `[dev]`: Testing and linting tools (`pytest`, `ruff`, `mypy`).
   - `[ml]`: TensorFlow support for ML components.
   - `[accel]`: Numba acceleration.

3. **Pre-commit**: Install pre-commit hooks to ensure code quality:
   ```powershell
   pre-commit install
   ```

## Testing Information

### Running Tests
The project uses `pytest` with specific markers to categorize tests.
- **Unit & Integration**: `python -m pytest -m "unit or integration"`
- **Diagnostic Tests**: `python -m pytest tests/diagnostic/test_comparison_setup.py` (Verify MATLAB/Python comparison environment).
- **All Tests**: `python -m pytest tests/`

### Adding New Tests
- **Location**: Place unit tests under `tests/unit/` mirroring the `source/slavv/` structure.
- **Markers**: Use `@pytest.mark.<type>` (e.g., `unit`, `integration`, `slow`, `regression`).
- **Numpy/SciPy Integration**: Most tests involve numerical data; use `np.array` for path data or volume mocks.

### Demonstration Process
To create and run a new test:
1. Create a file `tests/unit/test_feature_x.py`.
2. Implement your test function and mark it.
3. Run specifically: `python -m pytest tests/unit/test_feature_x.py`.

Example of a simple valid test:
```python
import numpy as np
import pytest
from slavv.utils import calculate_path_length

@pytest.mark.unit
def test_path_calculation():
    path = np.array([[0,0,0], [1,0,0]], dtype=float)
    assert calculate_path_length(path) == 1.0
```

## Additional Development Information

### Code Style & Linting
The project strictly follows `ruff` for formatting and linting.
- **Format**: `python -m ruff format source tests`
- **Lint**: `python -m ruff check source tests --fix`
- **Type Checking**: `python -m mypy` (Note: Type checking is being phased in gradually; focused on entry points).

### Architecture & Logging
- **Core Logic**: Main logic resides in `source/slavv/`.
- **Logging**: Use the standard `logging` module in library code. Avoid `print()` except in CLI output or apps.
- **MATLAB Parity**: When modifying core algorithms, ensure parity with the original MATLAB implementation. Use `workspace/scripts/cli/compare_matlab_python.py` to validate changes against MATLAB outputs if a local MATLAB installation is available.

### Utility Helpers
Windows developers can use `.\make.ps1` as a shortcut for common tasks like `install`, `format`, `lint`, and `test`.
