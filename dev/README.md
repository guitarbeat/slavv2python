# Development Workspace

This directory holds maintained developer-facing assets that support the
package source under `source/slavv/`.

## Directory Map

| Path | Purpose |
| --- | --- |
| `scripts/cli/` | MATLAB comparison entrypoints, wrapper launchers, and canonical parity parameters |
| `scripts/` | Utility scripts for repository maintenance and cleanup |
| `scratch/` | Temporary scratch files for quick experiments and checks (not maintained) |
| `tests/` | Unit, integration, UI, and diagnostic coverage for maintained package and workspace helper behavior |
| `tmp_tests/` | Temporary test artifacts (auto-managed by pytest, can be cleaned) |

## Start Here

- Use [tests/README.md](tests/README.md) for canonical test placement and scope.
- Use `scripts/cli/parity_experiment.py` for developer reruns, fail-fast exact-route gates, and exact artifact proof against reusable staged comparison roots.
- Use `../docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md` for the canonical source-of-truth hierarchy and parity claim boundaries.
- Use `../docs/reference/core/MATLAB_PARITY_MAPPING.md` when the task requires exact imported-MATLAB parity mapping or a Python-vs-MATLAB audit.

## Key Scripts

| Path | Purpose |
| --- | --- |
| `scripts/cli/parity_experiment.py` | Developer runner for fail-fast exact-route gates (`preflight-exact`, `prove-luts`, `capture-candidates`, `replay-edges`, `fail-fast`), reusable reruns, and exact artifact proof against preserved MATLAB vectors |
| `scripts/cleanup_deprecated_runs.ps1` | PowerShell utility to clean up deprecated run directories |

## Consolidation Notes

- Generated caches such as `__pycache__/` do not belong in `dev/` and can be
  removed safely.
- New developer utilities should live under `scripts/` instead of creating
  one-off top-level folders.
- New tests should follow ownership-based placement under `tests/`, not the
  task name that introduced them.
- The `scratch/` directory contains temporary experimental scripts that are not
  maintained and can be cleaned up periodically.
- Rich legacy parity diagnostics are not part of the live source tree. The maintained `parity_experiment.py` helper now covers fail-fast exact-route gates, rerun summaries, and exact artifact proof, but not the removed rich legacy diagnostics.
