# Development Workspace

This directory holds maintained developer-facing assets that support the
package source under `source/slavv/`.

## Directory Map

| Path | Purpose |
| --- | --- |
| `scripts/cli/` | MATLAB comparison entrypoints, wrapper launchers, and canonical parity parameters |
| `scripts/maintenance/` | Repository-maintenance helpers for MATLAB mapping and script audits |
| `scripts/benchmarks/` | Benchmark helpers for profiling visualization-heavy workflows |
| `tests/` | Unit, integration, UI, and diagnostic coverage for maintained package and workspace helper behavior |

## Start Here

- Use [tests/README.md](tests/README.md) for canonical test placement and scope.
- Use [scripts/maintenance/README.md](scripts/maintenance/README.md) for the maintained helper-script inventory.
- Use `scripts/cli/compare_matlab_python.py` for staged MATLAB/Python parity workflows.

## Consolidation Notes

- Generated caches such as `__pycache__/` do not belong in `dev/` and can be
  removed safely.
- New developer utilities should live under the existing `scripts/` buckets
  instead of creating one-off top-level folders.
- New tests should follow ownership-based placement under `tests/`, not the
  task name that introduced them.
