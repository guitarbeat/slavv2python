# Contributing

Thanks for your interest in contributing! This document outlines conventions for contributors and automated agents working on this repository.

## Documentation
- Place supplementary Markdown files in the `docs/` directory.
- Use relative links when referencing these documents from elsewhere in the repo.

## Code Style
- Follow basic PEP 8 formatting for Python code.
- **Type Hinting**: All public functions in `source/` must have Python type hints (e.g., `def func(x: int) -> float:`).
- **Docstrings**: Use reStructuredText (RST) or Google-style docstrings for all exported members.
- **Logging**: Use the standard `logging` module.
  - Do NOT use `print()` in library code (`source/`).
  - For scripts (`examples/`), use a consistent prefix for `print` messages (e.g., `[MyScript]: Message`).

## Programmatic Checks & Testing
- Compile check for all Python files:
  ```bash
  python -m compileall source/ tests/
  ```
- Run tests from the repo root (the `source` layout is automatically handled by `pyproject.toml` editable installs, or manual path setting):
  ```bash
  # Option 1: Install in editable mode (Recommended)
  pip install -e .
  pytest -q

  # Option 2: Manual PYTHONPATH
  export PYTHONPATH=$PYTHONPATH:$(pwd)/source
  pytest -q
  ```
  To run one test module:
  ```bash
  pytest -q tests/unit/test_energy.py
  ```

## Commit Messages
- Provide concise summaries of the changes made.

## Reducing Temp Files
The project is configured to minimize temp files when running:
- **Pytest**: Cache disabled (`-p no:cacheprovider`) — no `.pytest_cache/`
- **Python bytecode**: `setup_env.ps1` configures `PYTHONDONTWRITEBYTECODE=1` in venv/conda — no `__pycache__/`
- **Jupyter**: Optional config in `.jupyter/` stores checkpoints in system temp instead of each notebook dir. Use:
  ```powershell
  $env:JUPYTER_CONFIG_DIR = ".\.jupyter"; jupyter notebook
  ```
  Or for manual setup: `$env:PYTHONDONTWRITEBYTECODE = "1"` before running Python.

## Where to Start
- See [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md) for MATLAB→Python port status and mapping.
- See [docs/guides/DEVELOPMENT.md](docs/guides/DEVELOPMENT.md) for current status and roadmap.

## Regression Prevention (Critical)

### Golden Rules
1. **Never Break Existing Tests** — All PRs must pass `pytest tests/ -v`
2. **Lock Expected Outputs** — Use `np.allclose()` with tolerance 1e-6 for floats
3. **Preserve MATLAB Parity** — Check mapping doc before modifying core algorithms
4. **No Breaking Changes** — Adding optional params is OK; changing signatures is not

### Before Pushing Any Core Changes
Run the regression workflow:
```bash
pytest tests/ -v -x                        # Full suite, stop on first failure
pytest tests/ -v                           # Full suite
python examples/run_tutorial.py            # Quick integration check
```

Or invoke: `/regression-check`

### Adding New Regression Tests
```python
def test_my_feature_regression():
    input_data = create_synthetic_input()
    result = my_function(input_data)
    expected = np.array([...])  # Hardcoded expected values
    assert np.allclose(result, expected, atol=1e-6)
```
