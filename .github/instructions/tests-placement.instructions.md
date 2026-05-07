---
applyTo: "tests/**/*.py"
description: "Use when creating or modifying tests. Enforces ownership-based test placement, folder-derived markers, and repo-local tmp_path behavior."
---
# Test Authoring Instructions

## Placement

- Keep tests organized by owning surface, not task history.
- Use `tests/unit/<owner>/` for package-owned unit behavior (`analysis`, `apps`, `core`, `io`, `parity`, `runtime`, `utils`).
- Use `tests/unit/workspace_scripts/` for maintained helpers under `scripts/`.
- Use `tests/integration/` for cross-component workflows.
- Use `tests/ui/` for Streamlit and visualization-facing behavior.
- Use `tests/diagnostic/` for environment checks and MATLAB parity harness coverage.
- If a test is misfiled, move it to the matching owner directory instead of reshaping production code around location.

## Markers And Selection

- Do not hand-add folder markers (`unit`, `integration`, `ui`, `diagnostic`) when folder placement already conveys intent.
- `tests/conftest.py` auto-assigns markers by folder, and adds `regression` when `regression` appears in the node id.
- Keep regression intent explicit in test names and assertions when behavior is parity- or compatibility-sensitive.

## Temp Paths And Artifacts

- Use the repo-local `tmp_path` fixture from `tests/conftest.py`.
- Temporary test artifacts must stay under `workspace/tmp_tests/`, not system temp directories.
- Write repository-managed text fixtures/artifacts with explicit encodings, typically `encoding="utf-8"`.

## Keep In Sync

- Follow and link to `tests/README.md` for placement conventions.
- Follow `tests/conftest.py` for fixture and marker behavior.
- When tests touch comparison/parity behavior, preserve staged layout semantics documented in `docs/reference/workflow/PARITY_EXPERIMENT_STORAGE.md`.
