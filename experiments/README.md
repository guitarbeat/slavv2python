# Experiments Directory

This directory stores the outputs of various pipeline runs and comparisons, organized hierarchically for easy navigation.

## Directory Structure

Experiments are grouped by **Year** and **Month**, with descriptive subfolder names.

### Hierarchy
`experiments/YYYY/MM-MonthName/DD_HHMMSS_{Label}/`

- `YYYY`: Year of execution (e.g., `2026`)
- `MM-MonthName`: Month numeric and name (e.g., `02-February`)
- `DD_HHMMSS`: Day and timestamp
- `{Label}`: A descriptive label for the run (e.g., `Validation-Run`, `Parameter-Test`)

### Example Structure
```
experiments/
├── 2026/
│   ├── 01-January/
│   │   └── ...
│   └── 02-February/
│       ├── 09_173550_Baseline-Python-Run/
│       │   ├── checkpoints/
│       │   ├── results.json
│       │   └── MANIFEST.md
│       └── 10_101213_Low-Energy-Threshold-Test/
│           └── ...
```

## Creating Experiments

The pipeline automatically creates these folders when run via:

1. **Jupyter Notebooks**: `notebooks/01_End_to_End_Comparison.ipynb` allows you to set an `experiment_label` before running.
2. **Library**: The `slavv.evaluation.management.create_experiment_path` helper handles the folder creation logic.

## Workflow recommendation

1. **Run**: Execute your pipeline. A new nested folder is created based on the date and your provided label.
2. **Analyze**: Check the `results.json` and the auto-generated `MANIFEST.md` within the new folder.
3. **Compare**: Use `notebooks/01_End_to_End_Comparison.ipynb` or the specialized visualization notebooks to compare results between different experiment folders.
