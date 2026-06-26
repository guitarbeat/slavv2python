---
applyTo: "tests/**/*.py"
description: "Use when creating or modifying tests. Enforces ownership-based test placement, folder-derived markers, and repo-local tmp_path behavior."
---
# Test Authoring Instructions

## Placement

Tests mirror `slavv_python/` exactly — same folder name, same depth.

- Use `tests/unit/<owner>/` for package-owned unit behavior:
  - `pipeline` — Pipeline stage logic (energy, vertices, edges, network) → `slavv_python/pipeline/`
  - `analytics` — Analytics, parity, curation, metrics → `slavv_python/analytics/`
  - `engine` — Engine orchestration and state → `slavv_python/engine/`
  - `interface` — CLI, Streamlit interface behavior → `slavv_python/interface/`
  - `storage` — Storage loaders and exporters → `slavv_python/storage/`
  - `schema` — Data models → `slavv_python/schema/`
  - `utils` — Validation, math, formatting helpers → `slavv_python/utils/`
  - `visualization` — Plotting and rendering → `slavv_python/visualization/`
  - `workflows` — Pipeline orchestration, profiles → `slavv_python/workflows/`
  - `scripts` — Maintained helpers under `scripts/`
- Use `tests/integration/` for cross-component workflows and end-to-end pipeline behavior.
- Use `tests/integration/parity/` for parity-specific integration tests.
- Use `tests/ui/` for Streamlit and visualization-facing behavior.
- If a test is misfiled, move it to the matching owner directory instead of reshaping production code around location.

## Markers And Selection

- Do not hand-add folder markers (`unit`, `integration`, `ui`) when folder placement already conveys intent.
- `tests/conftest.py` auto-assigns markers by folder, and adds `regression` when `regression` appears in the node id.
- Keep regression intent explicit in test names and assertions when behavior is parity- or compatibility-sensitive.

## Temp Paths And Artifacts

- Use the repo-local `tmp_path` fixture from `tests/conftest.py`.
- Temporary test artifacts must stay under `workspace/scratch/tmp_tests/`, not system temp directories.
- Write repository-managed text fixtures/artifacts with explicit encodings, typically `encoding="utf-8"`.

## Keep In Sync

- Follow and link to `tests/README.md` for placement conventions.
- Follow `tests/conftest.py` for fixture and marker behavior.
- When tests touch comparison/parity behavior, preserve staged layout semantics documented in `docs/reference/workflow/PARITY_EXPERIMENT_STORAGE.md`.
