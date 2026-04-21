# Test Organization

Keep tests under `dev/tests/` organized by the owning package surface.

## Placement Rules

- `dev/tests/unit/<owner>/` for package behavior owned by `analysis`, `apps`, `core`, `io`, `runtime`, `utils`, `visualization`, or `workflows`
- `dev/tests/unit/workspace_scripts/` for maintained helper scripts under `dev/scripts/`
- `dev/tests/integration/` for cross-component workflows and end-to-end pipeline behavior
- `dev/tests/ui/` for Streamlit- and visualization-facing behavior
- `dev/tests/diagnostic/` for environment or setup diagnostics that still exist in the active product

## Notes

- Prefer moving a misfiled test into the matching owner directory instead of reshaping production code around the old location.
- Keep regression intent in test names, markers, and assertions.
- Do not create new parity- or MATLAB-specific test areas; those legacy surfaces have been removed.
