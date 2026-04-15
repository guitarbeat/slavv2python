---
description: "Use when changing MATLAB parity logic, comparison workflows, or run-state behavior in source/slavv/parity and source/slavv/runtime. Enforces staged layout and legacy checkpoint compatibility checks."
applyTo: "source/slavv/{parity,runtime}/**/*.py"
---
# Parity-Safe Change Instructions

## Scope

- Apply these rules when editing parity/comparison/runtime logic under `source/slavv/parity/` and `source/slavv/runtime/`.
- Keep diffs minimal and parity-focused; avoid unrelated refactors.

## Compatibility Requirements

- Preserve staged run layout semantics (`01_Input/`, `02_Output/`, `03_Analysis/`, `99_Metadata/`).
- Preserve compatibility with legacy flat checkpoint/run layouts where current code supports both.
- Do not silently rename or relocate staged artifacts without matching compatibility handling.

## Validation Expectations

- Run parity diagnostic coverage when parity/comparison behavior changes:
  - `python -m pytest tests/diagnostic/test_comparison_setup.py`
- Run at least the standard boundary-crossing gate when changes cross module boundaries:
  - `python -m pytest -m "unit or integration"`
- Use lint/type checks when editing Python modules:
  - `python -m ruff check source tests`
  - `python -m mypy`

## Implementation Guardrails

- Treat `source/slavv/parity/run_layout.py` and `source/slavv/runtime/run_state.py` as compatibility-critical surfaces.
- Favor additive compatibility shims over breaking format changes.
- Add deterministic regression tests for behavior changes that affect parity outputs or layout resolution.

## References

- `docs/reference/COMPARISON_LAYOUT.md`
- `source/slavv/parity/run_layout.py`
- `source/slavv/runtime/run_state.py`
- `tests/diagnostic/test_comparison_setup.py`
