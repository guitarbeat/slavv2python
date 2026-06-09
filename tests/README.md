# Test Organization

Keep tests under `tests/` organized by the owning package surface.

## Placement Rules

Tests mirror `slavv_python/` exactly — same folder name, same depth:

| Source module | Test location |
|:--------------|:--------------|
| `slavv_python/pipeline/` | `tests/unit/pipeline/` |
| `slavv_python/analytics/` | `tests/unit/analytics/` |
| `slavv_python/analytics/parity/` | `tests/unit/parity/` |
| `slavv_python/engine/` | `tests/unit/engine/` |
| `slavv_python/interface/` | `tests/unit/interface/` |
| `slavv_python/storage/` | `tests/unit/storage/` |
| `slavv_python/schema/` | `tests/unit/schema/` |
| `slavv_python/utils/` | `tests/unit/utils/` |
| `slavv_python/visualization/` | `tests/unit/visualization/` |
| `slavv_python/workflows/` | `tests/unit/workflows/` |

- `tests/integration/` for cross-component workflows and end-to-end pipeline behavior
- `tests/integration/parity/` for parity pre-gate integration tests (ADR 0009 tiers 1–2)
- `tests/ui/` for Streamlit- and visualization-facing behavior

## Notes

- Prefer moving a misfiled test into the matching owner directory instead of reshaping production code around the old location.
- Keep regression intent in test names, markers, and assertions.
- Reuse the shared builders under `tests/support/` when a test needs synthetic payloads, run snapshots, checkpoints, or reusable network fixtures.
- Keep exact-route parity tests under `tests/unit/parity/` (mirrors `slavv_python/analytics/parity/`).
- Do not create new task-history or workstream-specific test directories; place tests by owner surface, not by the temporary project that introduced them.
