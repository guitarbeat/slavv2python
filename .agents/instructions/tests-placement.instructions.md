---
applyTo: "tests/**/*.py"
description: "Use when creating or modifying tests. Enforces ownership-based test placement, folder-derived markers, and repo-local tmp_path behavior."
---
# Test Authoring Instructions

## Placement

- Keep tests organized by owning surface, not task history.
- Use `tests/unit/<owner>/` for package-owned unit behavior:
  - `core` — Processing stage logic (energy, vertices, edges, network)
  - `analysis` — Analytics, parity, curation, metrics
  - `apps` — CLI, Streamlit interface behavior
  - `io` — Storage loaders and exporters
  - `runtime` — Engine state, run tracking, snapshots
  - `models` — Schema and data models
  - `utils` — Validation, math, formatting helpers
  - `visualization` — Plotting and rendering
  - `workflows` — Pipeline orchestration, profiles
  - `scripts` — Maintained helpers under `scripts/`
- Use `tests/integration/` for cross-component workflows and end-to-end pipeline behavior.
- Use `tests/integration/parity/` for parity-specific integration tests.
- Use `tests/ui/` for Streamlit and visualization-facing behavior.
- If a test is misfiled, move it to the matching owner directory instead of reshaping production code around location.

> **Note:** Test directory names (e.g., `tests/unit/core/`) use a simplified owner convention and do not need to mirror the full package path (e.g., `slavv_python/processing/stages/`). The mapping is: `core` → `processing/stages/*`, `apps` → `interface/*`, `analysis` → `analytics/*`, `io` → `storage/*`, `runtime` → `engine/state/*`.

## Markers And Selection

- Do not hand-add folder markers (`unit`, `integration`, `ui`) when folder placement already conveys intent.
- `tests/conftest.py` auto-assigns markers by folder, and adds `regression` when `regression` appears in the node id.
- Keep regression intent explicit in test names and assertions when behavior is parity- or compatibility-sensitive.

## Temp Paths And Artifacts

- Use the repo-local `tmp_path` fixture from `tests/conftest.py`.
- Temporary test artifacts must stay under `tmp_tests/`, not system temp directories.
- Write repository-managed text fixtures/artifacts with explicit encodings, typically `encoding="utf-8"`.

## Keep In Sync

- Follow and link to `tests/README.md` for placement conventions.
- Follow `tests/conftest.py` for fixture and marker behavior.
- When tests touch comparison/parity behavior, preserve staged layout semantics documented in `docs/reference/workflow/PARITY_EXPERIMENT_STORAGE.md`.
