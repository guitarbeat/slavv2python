# Documentation

Maintained reference docs live here so the repository root can stay focused on
entry points and project metadata.

## Files

| File | Purpose |
| --- | --- |
| `MATLAB_MAPPING.md` | High-level map from upstream MATLAB modules to Python modules, with current parity notes |
| `COMPARISON_LAYOUT.md` | Canonical staged layout for generated MATLAB/Python comparison runs |
| `PARITY_FINDINGS_2026-03-27.md` | Verified findings from the fresh March 27, 2026 parity reruns and recommended next steps |
| `EDGE_PARITY_IMPLEMENTATION_PLAN.md` | Implementation plan for narrowing the remaining MATLAB-vs-Python edge and strand parity gap |

## Root-Level Docs

These docs stay at the repository root because they function as entry points or
project metadata:

| File | Purpose |
| --- | --- |
| `README.md` | Project overview, quick start, and common workflows |
| `CHANGELOG.md` | Notable recent development changes |
| `AGENTS.md` | Repository instructions for coding agents |

## Related Utilities

- `workspace/scripts/maintenance/check_mapped.py` updates the unmapped-file appendix in `MATLAB_MAPPING.md`.
- `workspace/scripts/maintenance/find_matlab_scripts.py` lists upstream MATLAB scripts that are not declared as functions.
- `workspace/reports/tooling/` stores archived linter and test-output snapshots that were moved out of the repository root.
