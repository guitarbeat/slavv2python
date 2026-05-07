# Test Organization

Keep tests under `workspace/tests/` organized by the owning package surface.

## Placement Rules

- `workspace/tests/unit/<owner>/` for package behavior owned by `analysis`, `apps`, `core`, `io`, `runtime`, `utils`, `visualization`, or `workflows`
- `workspace/tests/unit/workspace_scripts/` for maintained helper scripts under `workspace/scripts/`
- `workspace/tests/integration/` for cross-component workflows and end-to-end pipeline behavior
- `workspace/tests/ui/` for Streamlit- and visualization-facing behavior
- `workspace/tests/diagnostic/` for environment or setup diagnostics that still exist in the active product

## Notes

- Prefer moving a misfiled test into the matching owner directory instead of reshaping production code around the old location.
- Keep regression intent in test names, markers, and assertions.
- Reuse the shared builders under `workspace/tests/support/` when a test needs synthetic payloads, run snapshots, checkpoints, or reusable network fixtures.
- Keep exact-route parity tests under the owning package surface, or under `workspace/tests/unit/workspace_scripts/` for maintained developer runners such as `workspace/scripts/cli/parity_experiment.py`.
- Do not create new task-history or workstream-specific test directories; place tests by owner surface, not by the temporary project that introduced them.
