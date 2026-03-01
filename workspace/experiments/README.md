# Experiments Directory

This directory stores outputs from pipeline runs and comparisons.

## Directory Structure

Experiments are grouped by year and month:

`workspace/experiments/YYYY/MM-MonthName/DD_HHMMSS_{Label}/`

- `YYYY`: Year of execution (for example `2026`)
- `MM-MonthName`: Month number and name (for example `02-February`)
- `DD_HHMMSS`: Day and timestamp
- `{Label}`: Run label (for example `Validation-Run`)

### Example

```text
workspace/experiments/
  2026/
    02-February/
      09_173550_Baseline-Python-Run/
        checkpoints/
        results.json
        MANIFEST.md
```

## Creating Experiments

Experiment folders are created automatically when running:

1. `workspace/notebooks/01_End_to_End_Comparison.ipynb` (set `experiment_label`)
2. Library code via `slavv.evaluation.management.create_experiment_path`

## Recommended Workflow

1. Run pipeline to generate a timestamped folder.
2. Inspect `results.json` and `MANIFEST.md`.
3. Compare runs with `workspace/notebooks/01_End_to_End_Comparison.ipynb` or dashboard notebooks.
