---
description: "Use when changing MATLAB parity logic, comparison workflows, or run-state behavior in source/analysis/parity and source/runtime. Enforces staged layout compatibility checks."
applyTo: "source/{analysis/parity,runtime}/**/*.py"
---
# Parity-Safe Change Instructions

## Scope

- Apply these rules when editing parity/comparison/runtime logic under `source/analysis/parity/` and `source/runtime/`.
- Keep diffs minimal and parity-focused; avoid unrelated refactors.

## Compatibility Requirements

- Preserve staged run layout semantics (`01_Input/`, `02_Output/`, `03_Analysis/`, `99_Metadata/`).
- Do not silently rename or relocate staged artifacts without matching compatibility handling.

## Validation Expectations

- Run parity diagnostic coverage when parity/comparison behavior changes:
  - `python -m pytest dev/tests/diagnostic/test_comparison_setup.py`
- Run at least the standard boundary-crossing gate when changes cross module boundaries:
  - `python -m pytest -m "unit or integration"`
- Use lint/type checks when editing Python modules:
  - `python -m ruff check source dev/tests`
  - `python -m mypy`

## Implementation Guardrails

- Treat `source/analysis/parity/execution.py` and `source/runtime/run_state.py` as compatibility-critical surfaces.
- Favor additive compatibility shims over breaking format changes.
- Add deterministic regression tests for behavior changes that affect parity outputs or layout resolution.

## References

- `docs/reference/workflow/PARITY_EXPERIMENT_STORAGE.md`
- `source/analysis/parity/execution.py`
- `source/runtime/run_state.py`
- `dev/tests/diagnostic/test_comparison_setup.py`
