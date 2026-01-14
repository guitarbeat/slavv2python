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
pytest tests/test_regression_edges.py -v  # Critical regression test
pytest tests/ -v                           # Full suite
python examples/run_headless_demo.py       # Quick integration check
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
