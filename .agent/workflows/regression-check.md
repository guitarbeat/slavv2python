---
description: Workflow to verify no regressions before pushing changes
---

# Regression Check Workflow

Use this workflow before pushing any significant changes to ensure no regressions are introduced in the SLAVV project.

## Steps

// turbo
1. **Run Code Formatters and Linters:**
   ```bash
   ruff check .
   biome check .
   ```
   All checks must pass. If tools report fixable errors, proactively run `ruff check . --fix` and `biome check --write .` to resolve them automatically. Re-run the checks to confirm.

// turbo
2. **Run the Full Test Suite:**
   ```bash
   pytest tests/ -v
   ```
   All tests must pass. Find any failures or tracebacks and fix the underlying code before proceeding.

// turbo
3. **Run Demonstration Scripts:**
   ```bash
   python workspace/examples/run_tutorial.py
   ```
   Verify the script completes without raising any `Exception` or `RuntimeError`. This ensures the public-facing API is stable.

// turbo
4. **Check for New Deprecation Warnings:**
   ```bash
   python -W error::DeprecationWarning workspace/examples/run_tutorial.py
   ```
   If any deprecation warnings are thrown, resolve them by updating the deprecated API usage in the `source/slavv` directories.

## When to Use This Workflow
- Before creating a Pull Request or pushing to `main`.
- After modifying core algorithms inside the `source/slavv` package.
- After upgrading key mathematical or processing dependencies (such as NumPy, SciPy, scikit-image, or Numba).
