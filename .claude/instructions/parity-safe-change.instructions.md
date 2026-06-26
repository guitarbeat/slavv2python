---
description: "Use when changing MATLAB parity logic, comparison workflows, or run-state behavior in slavv_python/analytics/parity and slavv_python/engine/state. Enforces staged layout compatibility checks."
applyTo: "slavv_python/{analytics/parity,engine/state}/**/*.py"
---
# Parity-Safe Change Instructions

## Scope

- Apply these rules when editing parity/comparison/runtime logic under `slavv_python/analytics/parity/` and `slavv_python/engine/state/`.
- Keep diffs minimal and parity-focused; avoid unrelated refactors.

## Compatibility Requirements

- Preserve staged run layout semantics (`01_Input/`, `02_Output/`, `03_Analysis/`, `99_Metadata/`).
- Do not silently rename or relocate staged artifacts without matching compatibility handling.

## Validation Expectations

- Run parity-related tests when comparison behavior changes:
  - `python -m pytest tests/integration/parity/`
  - `python -m pytest tests/unit/analysis/ -k parity`
- Run the standard boundary-crossing gate when changes cross module boundaries:
  - `python -m pytest -m "unit or integration"`
- Use lint/type checks when editing Python modules:
  - `python -m ruff check slavv_python tests`
  - `python -m mypy`

## Implementation Guardrails

- Treat `slavv_python/analytics/parity/` and `slavv_python/engine/state/` as compatibility-critical surfaces.
- Favor additive compatibility shims over breaking format changes.
- Add deterministic regression tests for behavior changes that affect parity outputs or layout resolution.

## References

- [PARITY_CERTIFICATION_GUIDE.md](../../docs/reference/workflow/PARITY_CERTIFICATION_GUIDE.md)
- [WATERSHED_IMPLEMENTATION_NOTES.md](../../docs/reference/core/WATERSHED_IMPLEMENTATION_NOTES.md)
- [MATLAB_METHOD_IMPLEMENTATION_PLAN.md](../../docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md)
- `slavv_python/analytics/parity/` — Proof harness modules
- `slavv_python/engine/state/` — Run tracking and snapshot management
