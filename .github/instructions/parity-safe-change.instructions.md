---
description: "Use when changing MATLAB parity logic, comparison workflows, or run-state behavior in slavv_python/analysis/parity and slavv_python/runtime. Enforces staged layout compatibility checks."
applyTo: "slavv_python/{analysis/parity,runtime}/**/*.py"
---
# Parity-Safe Change Instructions

## Scope

- Apply these rules when editing parity/comparison/runtime logic under `slavv_python/analysis/parity/` and `slavv_python/runtime/`.
- Keep diffs minimal and parity-focused; avoid unrelated refactors.

## Compatibility Requirements

- Preserve staged run layout semantics (`01_Input/`, `02_Output/`, `03_Analysis/`, `99_Metadata/`).
- Do not silently rename or relocate staged artifacts without matching compatibility handling.

## Validation Expectations

- Run parity diagnostic coverage when parity/comparison behavior changes:
  - `python -m pytest workspace/tests/diagnostic/test_comparison_setup.py`
- Run at least the standard boundary-crossing gate when changes cross module boundaries:
  - `python -m pytest -m "unit or integration"`
- Use lint/type checks when editing Python modules:
  - `python -m ruff check source workspace/tests`
  - `python -m mypy`

## Implementation Guardrails

- Treat `slavv_python/analysis/parity/execution.py` and `slavv_python/runtime/run_state.py` as compatibility-critical surfaces.
- Favor additive compatibility shims over breaking format changes.
- Add deterministic regression tests for behavior changes that affect parity outputs or layout resolution.

## References

- `docs/reference/workflow/PARITY_EXPERIMENT_STORAGE.md`
- `slavv_python/analysis/parity/execution.py`
- `slavv_python/runtime/run_state.py`
- `workspace/tests/diagnostic/test_comparison_setup.py`
