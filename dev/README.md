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
- Use `scripts/cli/parity_experiment.py` for counts-only Python reruns against reusable staged comparison roots.

## Key Scripts

| Path | Purpose |
| --- | --- |
| `scripts/cli/parity_experiment.py` | Counts-only developer runner for copying reusable Python checkpoints into a fresh run root and rerunning from `edges` or `network` |
| `scripts/cli/run_matlab_vectorization.m` | MATLAB wrapper launched by the parity CLI |
| `scripts/maintenance/comparison_layout_smoothing.py` | Inventory legacy and grouped comparison runs, and refresh grouped archive metadata |
| `scripts/maintenance/refresh_matlab_mapping_appendix.py` | Refresh the generated MATLAB mapping appendix from upstream `.m` files |
| `scripts/maintenance/find_matlab_script_files.py` | Find `.m` files that behave like scripts instead of functions |
| `scripts/benchmarks/plot_2d_network_benchmark.py` | Manual timing probe for large 2D network plotting workloads |

## Consolidation Notes

- Generated caches such as `__pycache__/` do not belong in `dev/` and can be
  removed safely.
- New developer utilities should live under the existing `scripts/` buckets
  instead of creating one-off top-level folders.
- New tests should follow ownership-based placement under `tests/`, not the
  task name that introduced them.
- Rich legacy parity diagnostics are not part of the live source tree. The maintained `parity_experiment.py` helper is intentionally limited to preserved count truth plus fresh Python rerun counts.
