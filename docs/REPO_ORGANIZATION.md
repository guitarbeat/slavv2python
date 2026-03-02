# Repository Organization

This document defines where code, data, tests, and generated artifacts should live.

## Top-Level Layout

- `source/slavv/`: production package code only.
- `tests/`: all automated tests.
- `docs/`: developer and user documentation.
- `workspace/`: local workflows, notebooks, scripts, and generated local temp artifacts.
- `external/`: upstream or third-party external resources.

## Placement Rules

- New library code: `source/slavv/<domain>/...`
- New tests:
  - unit: `tests/unit/`
  - integration: `tests/integration/`
  - ui: `tests/ui/`
  - diagnostics/setup: `tests/diagnostic/`
- New developer docs: `docs/`
- New notebook or experiment helpers: `workspace/notebooks/` or `workspace/scripts/`

## Keep Out of Git

- Temporary test directories: `workspace/tmp_tests/`, `workspace/tmp_fixture/`, `tests/.pytest_tmp/`
- Local caches and generated logs/reports
- One-off ad-hoc outputs from experiments unless explicitly curated

## Testing Lanes

- Fast lane (default in CI): `pytest -m "unit or integration"`
- Full lane (nightly/manual): `pytest tests/`

Markers are auto-assigned by test folder in `tests/conftest.py`.

## Naming Conventions

- Tests: `test_<behavior>.py`
- Docs: descriptive `UPPER_SNAKE_CASE.md` or concise title case when already established
- Workspace scripts: verb-first names (for example, `run_*.py`, `setup_*.ps1`)

## Change Hygiene

- Keep production changes in `source/` separate from documentation-only updates when possible.
- If adding new folders, update this file and `docs/README.md`.
