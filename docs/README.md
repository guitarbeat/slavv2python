# Documentation

Maintained reference docs live here so the repository root can stay focused on
entry points and project metadata.

## Start Here

For rapid parity recall, start with:

1. [PARITY_HUB.md](PARITY_HUB.md)
2. [BOTTLENECK_TODO.md](../BOTTLENECK_TODO.md)
3. [EDGE_PARITY_IMPLEMENTATION_PLAN.md](EDGE_PARITY_IMPLEMENTATION_PLAN.md)

Recommended read pattern:

- use `PARITY_HUB.md` for the current state, default commands, and file map
- use `BOTTLENECK_TODO.md` for workflow and loop guidance
- use `EDGE_PARITY_IMPLEMENTATION_PLAN.md` for the active implementation target
- use `PARITY_FINDINGS_2026-03-27.md` when you need the evidence behind the
  current diagnosis

## Quick Map

| File | Purpose |
| --- | --- |
| `PARITY_HUB.md` | Fast recall hub for current parity status, default commands, and which file to open next |
| `MATLAB_MAPPING.md` | High-level map from upstream MATLAB modules to Python modules, with current parity notes |
| `COMPARISON_LAYOUT.md` | Canonical staged layout for generated MATLAB/Python comparison runs |
| `PARITY_FINDINGS_2026-03-27.md` | Verified findings from the fresh March 27, 2026 parity reruns and recommended next steps |
| `EDGE_PARITY_IMPLEMENTATION_PLAN.md` | Implementation plan for narrowing the remaining MATLAB-vs-Python edge and strand parity gap |
| `MATLAB_PARITY_AUDIT_CHECKLIST.md` | Check-off audit plan for comparing MATLAB and Python frontier-tracing semantics |

## By Question

| Question | Best file |
| --- | --- |
| What is true right now? | [PARITY_HUB.md](PARITY_HUB.md) |
| What should I run next? | [BOTTLENECK_TODO.md](../BOTTLENECK_TODO.md) |
| What is the active edge plan? | [EDGE_PARITY_IMPLEMENTATION_PLAN.md](EDGE_PARITY_IMPLEMENTATION_PLAN.md) |
| Why do we think the remaining blocker is in `edges`? | [PARITY_FINDINGS_2026-03-27.md](PARITY_FINDINGS_2026-03-27.md) |
| What exact MATLAB/Python behaviors should I audit next? | [MATLAB_PARITY_AUDIT_CHECKLIST.md](MATLAB_PARITY_AUDIT_CHECKLIST.md) |
| How do staged comparison outputs work? | [COMPARISON_LAYOUT.md](COMPARISON_LAYOUT.md) |
| How does the upstream MATLAB code map into Python? | [MATLAB_MAPPING.md](MATLAB_MAPPING.md) |

## Current Comparison Workflow Notes

- `docs/COMPARISON_LAYOUT.md` is the canonical reference for staged comparison
  outputs, including the shared `99_Metadata/` contract.
- The output-root preflight and MATLAB resume-transparency work has been folded
  into the maintained workflow docs and tests.
- `docs/EDGE_PARITY_IMPLEMENTATION_PLAN.md` tracks the implementation phases
  for narrowing the remaining parity gap.
- The stage-isolated MATLAB-edges-to-Python-network probe is now a supported
  workflow and is summarized in
  [stage_isolated_network_parity_2026-04-07.md](../workspace/reports/stage_isolated_network_parity_2026-04-07.md).

## Root-Level Docs

These docs stay at the repository root because they function as entry points or
project metadata:

| File | Purpose |
| --- | --- |
| `README.md` | Project overview, quick start, and common workflows |
| `CHANGELOG.md` | Notable recent development changes |
| `BOTTLENECK_TODO.md` | Maintained parity workflow and feedback-loop plan |
| `AGENTS.md` | Repository instructions for coding agents |

## High-Value Reports

These stay under `workspace/reports/`, but they are important enough to treat
as part of the parity doc surface:

| File | Purpose |
| --- | --- |
| `workspace/reports/stage_isolated_network_parity_2026-04-07.md` | Proof that exact MATLAB `edges` plus Python `network` can already converge exactly |
| `workspace/reports/python_matlab_parity_postfix_2026-03-30.md` | Historical parity checkpoint and context |
| `workspace/reports/python_standalone_consistency_postfix_2026-03-30.md` | Python repeatability context |

## Related Utilities

- `workspace/scripts/maintenance/check_mapped.py` updates the unmapped-file appendix in `MATLAB_MAPPING.md`.
- `workspace/scripts/maintenance/find_matlab_scripts.py` lists upstream MATLAB scripts that are not declared as functions.
- `workspace/reports/tooling/` stores archived linter and test-output snapshots that were moved out of the repository root.
