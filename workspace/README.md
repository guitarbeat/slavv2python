# Data Workspace

This directory is the canonical untracked home for experiment data, oracles, 
and results managed by the `scripts/parity_experiment.py` runner.

## Directory Map

| Path | Purpose |
| --- | --- |
| `datasets/` | Canonical untracked dataset manifests keyed by dataset hash |
| `oracles/` | Canonical untracked MATLAB oracle packages (raw vectors, normalized payloads) |
| `reports/` | Durable home for promoted parity reports |
| `runs/` | Disposable working space for developer runs and parity reruns |
| `index.jsonl` | Append-only catalog of all assets in this workspace |

## Storage Strategy

- **Disposable**: Treat `runs/` as ephemeral.
- **Durable**: Promote important results to `reports/` using `scripts/parity_experiment.py promote-report`.
- **Integrity**: Use `index.jsonl` to resolve assets by ID or hash.

## Related Tools

- Use `scripts/parity_experiment.py` for all run management and parity validation.
- Use `tests/` for validating both the core library and these workspace utilities.
