# Contributing

Thanks for your interest in contributing! This document outlines conventions for contributors and automated agents working on this repository.

## Documentation
- Place supplementary Markdown files in the `docs/` directory.
- Use relative links when referencing these documents from elsewhere in the repo.

## Code Style
- Follow basic PEP 8 formatting for Python code.
- **Type Hinting**: All public functions in `src/` must have Python type hints (e.g., `def func(x: int) -> float:`).
- **Docstrings**: Use reStructuredText (RST) or Google-style docstrings for all exported members.
- **Logging**: Use the standard `logging` module.
  - Do NOT use `print()` in library code (`src/`).
  - For scripts (`examples/`), use a consistent prefix for `print` messages (e.g., `[MyScript]: Message`).

## Programmatic Checks & Testing
- Compile check for all Python files:
  ```bash
  python -m compileall src/ tests/
  ```
- Run tests from the repo root (the `src` layout is automatically handled by `pyproject.toml` editable installs, or manual path setting):
  ```bash
  # Option 1: Install in editable mode (Recommended)
  pip install -e .
  pytest -q

  # Option 2: Manual PYTHONPATH
  export PYTHONPATH=$PYTHONPATH:$(pwd)/src
  pytest -q
  ```
  To run one test module:
  ```bash
  pytest -q tests/unit/test_energy.py
  ```

## Commit Messages
- Provide concise summaries of the changes made.

## Where to Start
- See `docs/MATLAB_TO_PYTHON_MAPPING.md` for canonical porting status (includes coverage and deviations).
- See `docs/ROADMAP.md` for planned work.

## Regression Prevention (Critical)

### Golden Rules
1. **Never Break Existing Tests** — All PRs must pass `pytest tests/ -v`
2. **Lock Expected Outputs** — Use `np.allclose()` with tolerance 1e-6 for floats
3. **Preserve MATLAB Parity** — Check mapping doc before modifying core algorithms
4. **No Breaking Changes** — Adding optional params is OK; changing signatures is not

### Before Pushing Any Core Changes
Run the regression workflow:
```bash
pytest tests/unit/test_regression.py -v    # Critical regression test
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
