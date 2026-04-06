# SLAVV Development Guidelines

This document provides project-specific guidelines for advanced developers working on the `slavv2python` repository.

## Build/Configuration Instructions

### Environment Setup

1.  **Virtual Environment**: Create and activate a Python virtual environment (Python 3.9+ recommended).
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
2.  **Installation**: Install the package with the necessary extras. For development and UI support:
    ```powershell
    pip install -e ".[app,dev]"
    ```
    Extras include:
    - `[app]`: Streamlit-based web interface (`slavv-app`).
    - `[dev]`: Testing and linting tools (`pytest`, `ruff`, `mypy`).
3.  **Pre-commit**: Install pre-commit hooks to ensure code quality:
    ```powershell
    pre-commit install
    ```

### Windows Helper
Windows developers can use `.\make.ps1` as a shortcut for common tasks:
```powershell
.\make.ps1 install
.\make.ps1 format
.\make.ps1 lint
.\make.ps1 test
```

## Testing Information

### Running Tests
The project uses `pytest` with specific markers to categorize tests.
- **Unit & Integration**: `python -m pytest -m "unit or integration"`
- **Diagnostic Tests**: `python -m pytest tests/diagnostic/test_comparison_setup.py` (Verify MATLAB/Python comparison environment).
- **All Tests**: `python -m pytest tests/`

### Adding and Executing New Tests
- **Location**: Place unit tests under `tests/unit/` mirroring the `source/slavv/` structure.
- **Markers**: Use `@pytest.mark.<type>` (e.g., `unit`, `integration`, `slow`, `regression`).
- **Numpy Integration**: Most tests involve numerical data; use `np.array` for path data or volume mocks.

### Testing Demonstration
To verify the testing environment and demonstrate the process, you can create a simple test for `slavv.utils.calculate_path_length`.

1.  **Create a test file** (e.g., `tests/unit/test_demo.py`):
    ```python
    import pytest
    from slavv.utils import calculate_path_length
    import numpy as np

    @pytest.mark.unit
    def test_demo_path_length():
        """A simple demo test to verify the testing environment."""
        # (0,0,0) -> (3,0,0) -> (3,4,0) => Expected length = 7.0
        path = np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0]], dtype=float)
        assert calculate_path_length(path) == 7.0
    ```

2.  **Run the test**:
    ```powershell
    python -m pytest tests/unit/test_demo.py
    ```

## Additional Development Information

### Code Style & Quality
The project strictly follows `ruff` for formatting and linting, and `mypy` for type checking.
- **Format**: `python -m ruff format source tests`
- **Lint**: `python -m ruff check source tests --fix`
- **Type Checking**: `python -m mypy`

### Architecture & Best Practices
- **Package Core**: Library code resides in `source/slavv/`.
- **Logging**: Use the standard `logging` module in library code. Avoid `print()` except in CLI output or apps.
- **MATLAB Parity**: When modifying core algorithms, maintain parity with the original MATLAB implementation. Use `workspace/scripts/cli/compare_matlab_python.py` for validation.
- **CLI Workflows**: Use `slavv run`, `slavv analyze`, and `slavv plot` for processing and visualization.
- **App Launcher**: Use `slavv-app` or `python -m streamlit run source/slavv/apps/web_app.py` for the UI.

### Guardrails
- Keep package code under `source/slavv/` and tests under `tests/`.
- Do not treat generated outputs under `comparisons/` or `comparison_output*/` as source inputs.
- Prefer searching with `rg`, but exclude noisy generated trees like `workspace/tmp_tests/`.
