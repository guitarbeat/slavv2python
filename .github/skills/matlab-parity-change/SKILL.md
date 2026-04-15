---
name: matlab-parity-change
description: "Implement MATLAB parity or comparison behavior changes safely. Use for run_layout/run_state updates, import-matlab behavior, staged-vs-legacy compatibility, and parity regression hardening."
argument-hint: "Describe the parity behavior to change and expected compatibility constraints"
user-invocable: true
---
# MATLAB Parity Change

## When To Use

- Changing MATLAB parity behavior in `source/slavv/parity/`, `source/slavv/runtime/`, or parity-sensitive CLI/app paths.
- Updating staged comparison layout handling.
- Adjusting legacy checkpoint compatibility behavior.
- Fixing parity regressions found in diagnostic comparisons.

## Goals

- Keep staged layout semantics stable.
- Preserve legacy compatibility where supported.
- Make the smallest safe diff.
- Add deterministic tests that lock in behavior.

## Procedure

1. Confirm scope and invariants
- Identify whether the change affects layout resolution, metadata, checkpoint compatibility, or report output.
- Record required invariants from `source/slavv/parity/run_layout.py` and `source/slavv/runtime/run_state.py`.

2. Inspect existing behavior
- Read nearest diagnostic/integration tests first.
- Trace call sites that consume layout/run-state outputs.
- Prefer link-first references to docs rather than duplicating rules.

3. Implement minimal change
- Apply the narrowest change that satisfies the requested behavior.
- Avoid opportunistic refactors in parity-critical paths.

4. Add or update deterministic tests
- Place tests by ownership under `tests/`.
- Cover both staged layout and legacy-layout compatibility when behavior touches either path.

5. Run validation gate
- `python -m pytest tests/diagnostic/test_comparison_setup.py`
- `python -m pytest -m "unit or integration"`
- `python -m ruff check source tests`
- `python -m mypy`

6. Report compatibility impact
- State what changed for staged layout behavior.
- State what remains compatible for legacy layouts.
- Identify residual risks and follow-up checks.

## Checklist

- [ ] Staged layout semantics preserved
- [ ] Legacy compatibility behavior preserved or explicitly documented
- [ ] Diagnostic parity test coverage updated
- [ ] Unit/integration boundary gate executed
- [ ] Lint and type checks completed
