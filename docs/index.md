# Documentation

Use this page as a starting point to navigate the repository’s docs.

## User Guide
- Quickstart and app usage: see the project `README.md` and `slavv-streamlit/README.md`.
- Public API reference: see `docs/PUBLIC_API.md` for functions and classes with examples and links to source.

## MATLAB Porting
- MATLAB → Python mapping with parity levels: `docs/MATLAB_TO_PYTHON_MAPPING.md`
- Coverage report of unmapped MATLAB files: `docs/MATLAB_COVERAGE_REPORT.md`
- Porting summary and major changes: `docs/PORTING_SUMMARY.md`
- Known parity deviations and rationale: `docs/PARITY_DEVIATIONS.md`

## Development
- Testing guide: `docs/TESTING.md`
- Contributing guidelines: `CONTRIBUTING.md`

## Helpful Notes
- Axis order is `(y, x, z)` throughout, and physical units use `microns_per_voxel`.
- Many MATLAB scripts (e.g., `vectorization_script_*`) are intentionally not ported; the app and tests cover equivalent workflows.

