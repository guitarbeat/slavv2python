# Development Workspace

This directory holds maintained developer-facing assets that support the
package source under `source/`.

## Directory Map

| Path | Purpose |
| --- | --- |
| `datasets/` | Canonical untracked dataset manifests keyed by dataset hash for parity bookkeeping |
| `oracles/` | Canonical untracked MATLAB oracle packages with preserved raw vectors, normalized payloads, and hashes |
| `reports/` | Canonical untracked home for promoted parity reports that should outlive disposable runs |
| `scripts/cli/` | MATLAB comparison entrypoints, wrapper launchers, and canonical parity parameters |
| `scripts/` | Utility scripts for repository maintenance and cleanup |
| `runs/` | Canonical untracked home for disposable developer run roots and parity reruns |
| `scratch/` | Temporary scratch files for quick experiments and checks (not maintained) |
| `tests/` | Unit, integration, UI, and diagnostic coverage for maintained package and workspace helper behavior |
| `tmp_tests/` | Untracked temporary test artifacts (auto-managed by pytest, can be cleaned) |

## Start Here

- Use [tests/README.md](tests/README.md) for canonical test placement and scope.
- Use `scripts/cli/parity_experiment.py` for developer reruns, fail-fast exact-route gates, oracle promotion, report promotion, and exact artifact proof.
- Use `../docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md` for the canonical source-of-truth hierarchy and parity claim boundaries.
- Use `../docs/reference/core/MATLAB_PARITY_MAPPING.md` when the task requires exact imported-MATLAB parity mapping or a Python-vs-MATLAB audit.
- Use `../docs/reference/workflow/PARITY_EXPERIMENT_STORAGE.md` for the maintained experiment-root layout under `datasets/`, `oracles/`, `runs/`, and `reports/`.

## Key Scripts

| Path | Purpose |
| --- | --- |
| `scripts/cli/parity_experiment.py` | Developer runner for fail-fast exact-route gates, reusable reruns, oracle promotion, report promotion, and exact artifact proof against preserved MATLAB vectors |

## Storage Notes

- Use one experiment root with sibling `datasets/`, `oracles/`, `runs/`, and `reports/` folders.
- Treat `runs/` as disposable working space and keep long-lived artifacts only after `promote-report`.
- Use `01_Params/shared_params.json`, `01_Params/python_derived_params.json`, and `01_Params/param_diff.json` as the maintained exact-route fairness surface.
- Use `03_Analysis/normalized/` plus `.sha256` sidecars for cheap proof comparisons against preserved MATLAB truth.

## Consolidation Notes

- Generated caches such as `__pycache__/` do not belong in `dev/` and can be
  removed safely.
- New developer utilities should live under `scripts/` instead of creating
  one-off top-level folders.
- Manual reruns and parity experiments should live under `runs/` and be treated
  as disposable unless promoted into a durable `reports/` entry.
- New tests should follow ownership-based placement under `tests/`, not the
  task name that introduced them.
- The `scratch/` directory contains temporary experimental scripts that are not
  maintained and can be cleaned up periodically.
- Rich legacy parity diagnostics are not part of the live source tree. The maintained `parity_experiment.py` helper now covers fail-fast exact-route gates, rerun summaries, and exact artifact proof, but not the removed rich legacy diagnostics.
