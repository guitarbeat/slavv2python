---
description: "Implement a Python code change with a minimal diff, add/update ownership-aligned tests, and run the standard validation gate."
name: "Implement Python Change"
argument-hint: "Change request + target files/modules + constraints"
agent: "agent"
---
Implement the requested Python code change in this repository with the smallest safe diff.

## Inputs
Use the user-provided arguments as the source of truth:
- What behavior must change
- Where to change it (files/modules if known)
- Constraints (backward compatibility, parity expectations, performance, etc.)

## Required Workflow
1. Locate the relevant code paths and existing tests before editing.
2. Apply a minimal, focused implementation change.
3. Add or update tests in ownership-aligned test locations under `dev/tests/`.
4. Run and report the standard validation gate:
   - `python -m ruff format --check source dev/tests`
   - `python -m ruff check source dev/tests`
   - `python -m mypy`
   - `python -m pytest -m "unit or integration"`
5. If the change touches parity/runtime behavior, also run relevant parity checks (for example `dev/tests/diagnostic/test_comparison_setup.py`) and preserve staged-layout expectations.
6. If a command fails, attempt the smallest practical fix related to the requested change, then re-run the affected checks.

## Output Format
Return the result in this structure:
1. Summary of behavior change
2. Files changed with one-line purpose each
3. Test changes and why they cover the behavior
4. Validation results (pass/fail per command)
5. Remaining risks or follow-ups

## Guardrails
- Preserve existing public APIs unless the request explicitly changes them.
- Do not include unrelated refactors.
- Prefer deterministic tests and avoid brittle timing-based assertions.
- Keep logging/CLI conventions and repository style consistent with surrounding code.

