# Workspace Guide

This file documents the `workspace/` area used for setup, execution, analysis, and local artifacts.

## Layout

- `workspace/scripts/`: automation and CLI wrappers
- `workspace/notebooks/`: interactive comparison and analysis notebooks
- `workspace/examples/`: runnable programmatic usage examples
- `workspace/experiments/`: generated run outputs
- `workspace/tmp_tests/`, `workspace/tmp_fixture/`: temporary local test artifacts

## Scripts

### `workspace/scripts/cli/`

- `compare_matlab_python.py`: run MATLAB/Python comparison workflows
- `comparison_params.json`: parameter file for comparison runs
- `run_matlab_cli.bat`, `run_matlab_cli.sh`: platform wrappers
- `run_matlab_vectorization.m`: MATLAB entry point

### `workspace/scripts/setup/`

- `setup_env.ps1`: environment setup helper
- `verify_imports.py`: quick dependency/import validation

## Notebooks

Primary flow:

1. `00_Setup_and_Validation.ipynb`
2. `01_Run_Matlab.ipynb`
3. `02_Run_Python.ipynb`
4. `03_Compare_Results.ipynb`
5. `04_Comparison_Dashboard.ipynb`

Additional analysis notebooks:

- `05_Statistical_Analysis.ipynb`
- `06_Data_Management.ipynb`
- `07_Tutorial.ipynb`

## Experiments Output Convention

Generated outputs are grouped by timestamped folders:

`workspace/experiments/YYYY/MM-MonthName/DD_HHMMSS_{Label}/`

Typical contents include `results.json`, `MANIFEST.md`, and optional `checkpoints/`.

## Hygiene

- Treat `workspace/tmp_tests/` and `workspace/tmp_fixture/` as disposable local artifacts.
- Keep curated outputs under `workspace/experiments/`; avoid committing ad-hoc scratch outputs.
