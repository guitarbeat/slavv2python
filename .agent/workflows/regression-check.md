---
description: Workflow to verify no regressions before pushing changes
---

# Regression Check Workflow

Use this workflow before pushing any significant changes to ensure no regressions are introduced.

## Steps

// turbo
1. **Run the full test suite:**
   ```bash
   pytest tests/ -v
   ```
   All tests must pass. If any fail, fix them before proceeding.

// turbo
2. **Run the regression test specifically:**
   ```bash
   pytest tests/test_regression_edges.py -v
   ```
   This test locks the expected output of edge tracing on synthetic data.

3. **If modifying core algorithms, run the tutorial on synthetic data:**
   ```bash
   python examples/run_headless_demo.py
   ```
   Verify it completes without errors.

4. **Check for new deprecation warnings:**
   ```bash
   python -W error::DeprecationWarning examples/run_headless_demo.py
   ```

## When to Use

- Before any PR that touches `vectorization_core.py`
- Before any PR that touches `ml_curator.py`
- After upgrading dependencies (numpy, scipy, scikit-image)

## Adding New Regression Tests

When adding a new feature:
1. Create a test in `tests/` that captures the expected output
2. Use `np.allclose()` for floating-point comparisons
3. Consider adding a fixture file for complex expected outputs
