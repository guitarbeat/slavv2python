---
name: python-change-gate
description: "Implement a Python change with the smallest safe diff, add/update ownership-aligned tests, and run the repo's standard validation gate (ruff, mypy, pytest -m 'unit or integration'). Use when: fixing bugs, refactoring, adding features, or touching CLI/app/runtime/parity code. Keywords: minimal diff, tests, pytest, ruff, mypy, validation gate."
argument-hint: "Describe the change + where it lives; this skill will drive implementation, tests, and validation."
user-invocable: true
---

# Python Change Gate

Codifies the repository's default workflow for making a Python change safely: minimal diff, tests that prove behavior, and the standard validation gate.

## When To Use
- Bug fixes, refactors, or feature work in `source/slavv/`
- CLI/app changes under `source/slavv/apps/`
- Runtime/run-state or parity-sensitive changes
- Any change where you want a consistent “done” definition

## Procedure

### 1) Scope and Read First
1. Identify the smallest set of modules impacted.
2. Read the nearest tests first (or create them in the ownership-aligned location per `tests/README.md`).
3. Confirm whether behavior is parity-sensitive (run layout, legacy checkpoints, MATLAB import).

### 2) Implement Minimal Diff
1. Prefer the smallest change that fixes the root cause.
2. Avoid unrelated formatting churn or broad refactors.
3. Preserve public APIs unless the task explicitly requires changes.

### 3) Add/Update Tests
1. Add tests that fail before the change and pass after.
2. Keep tests deterministic (avoid time/randomness unless controlled).
3. Place tests under the owning surface area (see `tests/README.md`).

### 4) Validate (Standard Gate)
Run these from repo root:

```powershell
python -m ruff check source tests
python -m ruff format --check source tests
python -m mypy
python -m pytest -m "unit or integration"
```

If the change is parity/comparison related, also run:

```powershell
python -m pytest tests/diagnostic/test_comparison_setup.py
```

### 5) Report Results
Return:
1. Files changed (one-line purpose each).
2. Tests added/updated and what they cover.
3. Commands run and pass/fail summary.
4. Any remaining risks, assumptions, or follow-ups.

## Completion Criteria
- Behavior change is covered by tests (or explicitly justified if not feasible).
- Standard gate passes (or failures are explained and scoped).
- No unrelated churn; diff is reviewable.
