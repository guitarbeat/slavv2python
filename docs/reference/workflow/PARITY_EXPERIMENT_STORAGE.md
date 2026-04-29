# Parity Experiment Storage

[Up: Reference Docs](../README.md)

This guide defines the maintained storage model for developer MATLAB-parity work.
Use it when preparing preserved MATLAB truth, running disposable Python parity
trials, or promoting durable summaries.

## Root Layout

Use a dedicated experiment root such as:

```text
D:\slavv_comparisons\experiments\live-parity\
  datasets\
  oracles\
  runs\
  reports\
  index.jsonl
```

- `datasets/` holds immutable input volumes or dataset packages plus dataset
  manifests and hashes.
- `oracles/` holds preserved MATLAB truth packages only.
- `runs/` holds disposable Python reruns and developer fail-fast artifacts.
- `reports/` holds promoted summaries copied out of disposable runs.
- `index.jsonl` is the append-only catalog for runs, oracles, and promoted
  reports.

## Run Layout

Each parity run should use this structure:

```text
runs\<run_id>\
  00_Refs\
  01_Params\
  02_Output\
  03_Analysis\
  99_Metadata\
```

- `00_Refs/` stores copied source references such as prior comparison reports,
  source run snapshots, source run manifests, and oracle manifests.
- `01_Params/` stores:
  - `shared_params.json`
  - `python_derived_params.json`
  - `param_diff.json`
- `02_Output/` stores Python checkpoints and stage-owned artifacts.
- `03_Analysis/` stores proof reports, normalized proof payloads, and `.sha256`
  sidecars.
- `99_Metadata/` stores `run_snapshot.json`, `run_manifest.json`, and
  command-specific provenance files.

## Oracle Layout

Each MATLAB oracle should be promoted into its own root under `oracles/`:

```text
oracles\<oracle_id>\
  01_Input\matlab_results\<batch_id>\
  03_Analysis\normalized\
  99_Metadata\oracle_manifest.json
```

The oracle manifest should record at least:

- `oracle_id`
- `dataset_hash`
- `matlab_source_version`
- raw vector paths and hashes
- normalized artifact paths
- timestamps and retention policy

## Dataset Layout

Each reusable input volume should be promoted into its own root under `datasets/`:

```text
datasets\<dataset_hash>\
  01_Input\<filename>.tif
  99_Metadata\dataset_manifest.json
```

The dataset manifest should record at least:

- `dataset_hash`
- stored input path
- original source path
- input byte size
- timestamps and retention policy

## Parameter Separation

Exact-route runs should split settings into:

- `shared_params.json`: settings that must match the MATLAB method surface
- `python_derived_params.json`: orchestration, parity-only, and other
  Python-owned settings
- `param_diff.json`: required exact values, Python-only controls, unclassified
  keys, and content hashes for the split payloads

This separation is the maintained fairness surface for deciding whether a run is
actually comparing the same method.

## Retention Rules

- Treat everything under `runs/` as disposable.
- Use `promote-report` to copy analysis artifacts into `reports/` when a run
  should be kept for discussion or recordkeeping.
- Treat `oracles/` as preserved source material, not scratch space.

## Canonical Commands

Promote a MATLAB batch into an oracle:

```powershell
python dev/scripts/cli/parity_experiment.py promote-dataset `
    --dataset-file C:\path\to\volume.tif `
    --experiment-root D:\slavv_comparisons\experiments\live-parity

python dev/scripts/cli/parity_experiment.py promote-oracle `
    --matlab-batch-dir D:\incoming\batch_260421-151654 `
    --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a `
    --dataset-file D:\datasets\volume.tif `
    --oracle-id v22_a
```

Run a disposable native-first trial:

```powershell
python dev/scripts/cli/parity_experiment.py preflight-exact `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run `
    --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\trial_b
```

Promote a kept summary:

```powershell
python dev/scripts/cli/parity_experiment.py promote-report `
    --run-root D:\slavv_comparisons\experiments\live-parity\runs\trial_b
```

## Notes

- `prove-exact` still compares normalized Python checkpoints against preserved
  MATLAB vectors.
- The storage model does not change claim boundaries: MATLAB source and
  preserved MATLAB vectors remain the proof authority.
