# SLAVV Python Port

Python reimplementation of **SLAVV** (Segmentation-Less, Automated, Vascular Vectorization) for 3D vascular network extraction from microscopy volumes. This repository contains the core package code, a Streamlit web application, a Command-Line Interface (CLI), MATLAB import helpers, and parity/comparison tooling used to validate the port against the original MATLAB implementation.

## Overview

SLAVV provides an automated pipeline to extract vascular graphs (vertices and edges) directly from 3D image data without an intermediate segmentation step. This Python port aims for high parity with the original MATLAB implementation while providing a modern, scalable, and easy-to-use Python interface.

## Stack

- **Language**: Python 3.9+
- **Numerical/Scientific**: NumPy, SciPy, scikit-image, scikit-learn
- **Graph Processing**: NetworkX
- **I/O**: Tifffile, h5py, Pandas, Pillow
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Acceleration**: Numba (optional via `[accel]`)
- **Frameworks**: Streamlit (Web App), argparse (CLI)
- **Tooling**: Ruff (formatting/linting), MyPy (type checking), Pytest (testing)
- **Package Manager**: `pip` (with `setuptools`)

## Requirements

- **Python**: 3.9, 3.10, 3.11, or 3.12.
- **MATLAB** (Optional): Required only for parity comparisons and MATLAB result imports. Recommended version: R2019a or newer.

## Setup

1. **Create and activate a virtual environment**:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install the package**:

   Choose the installation extra that matches your needs:

   ```powershell
   # Basic installation (CLI only)
   pip install -e .

   # Installation with the Streamlit app
   pip install -e ".[app]"

   # Full developer installation (includes tests, linting, and app)
   pip install -e ".[app,dev]"

   # Other available extras: [ml, notebooks, dicom, sitk, cupy, zarr, napari, accel, all]
   ```

3. **Install pre-commit hooks** (recommended for contributors):

   ```powershell
   pre-commit install
   ```

## Entry Points

### Command-Line Interface (CLI)

Use the `slavv` command for headless or scripted workflows:

```powershell
# Get system and version information
slavv info

# Run the full pipeline on a TIFF volume
slavv run -i volume.tif -o slavv_output --export csv json

# Run with custom parameters and an explicit structured run directory
slavv run -i volume.tif -o slavv_output --run-dir dev\runs\sample_a --vessel-radius 2.0

# Import MATLAB results for comparison
slavv import-matlab -b path\to\matlab_batch -c my_checkpoints

# Analyze and plot results from the standard JSON export
slavv analyze -i slavv_output/network.json
slavv plot -i slavv_output/network.json -o plots.html
```

### Web Application

After installing with the `[app]` extra, start the interactive UI:

```powershell
slavv-app
# Or manually via streamlit:
python -m streamlit run source/slavv/apps/web_app.py
```

### MATLAB Parity And Comparison Workflow

Use the backward-compatible wrapper at `dev/scripts/cli/compare_matlab_python.py` for MATLAB/Python parity loops.

```powershell
# Default parity run
python dev/scripts/cli/compare_matlab_python.py `
  --input data/slavv_test_volume.tif `
  --matlab-path "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" `
  --output-dir comparison_output
```

See [AGENTS.md](AGENTS.md) for more advanced comparison flags (`--skip-matlab`, `--resume-latest`, etc.).

## Scripts and Maintenance

Maintenance and utility scripts are located in `dev/scripts/`:

- **CLI Wrappers**:
  - `dev/scripts/cli/compare_matlab_python.py`: Backward-compatible wrapper for parity validation workflows.
- **Maintenance**:
  - `dev/scripts/maintenance/find_matlab_script_files.py`: Audits and locates MATLAB script dependencies.
  - `dev/scripts/maintenance/refresh_matlab_mapping_appendix.py`: Updates documentation mapping between MATLAB and Python symbols.
  - `dev/scripts/maintenance/comparison_layout_smoothing.py`: Normalizes and migrates legacy comparison layouts to the staged structure.
- **Benchmarks**:
  - `dev/scripts/benchmarks/plot_2d_network_benchmark.py`: Performance visualization for 2D network extraction.

No repo-local wrapper scripts are used for formatting, linting, or tests. Use the canonical `python -m ruff`, `python -m mypy`, and `python -m pytest` commands instead.

## Environment Variables

- `PYTHONUTF8`: Set to `1` by `slavv-app` for UTF-8 support on Windows.
- `PYTHONIOENCODING`: Set to `utf-8` by `slavv-app`.
- `SLAVV_CHECKPOINT_DIR`: (Optional) Default directory for legacy flat checkpoints.
- TODO: Document any other environment variables used for resource management or logging levels.

## Testing

Run tests using `pytest`:

```powershell
# Run unit and integration tests (recommended)
python -m pytest -m "unit or integration"

# Diagnostic tests for environment/MATLAB parity setup
python -m pytest dev/tests/diagnostic/test_comparison_setup.py

# Full suite (includes slow regression and UI tests)
python -m pytest
```

## Project Structure

| Path | Description |
| --- | --- |
| `source/slavv/` | Core package code (processing, I/O, analysis, visualization, CLI/app). |
| `dev/tests/` | Unit, integration, UI, regression, and diagnostic tests. |
| `dev/scripts/` | MATLAB comparison wrappers, maintenance helpers, and benchmarks. |
| `docs/` | Reference docs: translation guide, MATLAB mapping, and comparison layout. |
| `external/` | Optional local checkouts (e.g., `Vectorization-Public` for MATLAB SLAVV). |
| `data/` | Sample data and test volumes. |
| `slavv_comparisons/`| Default root for structured comparison outputs. |

## License

This project is licensed under the **GNU GPL-3.0**. See the `LICENSE` file for details.

## Documentation and TODOs

For detailed contributor guidelines, see [AGENTS.md](AGENTS.md).
For the current parity backlog and roadmap, see [TODO.md](TODO.md).
For a list of recent changes, see [CHANGELOG.md](CHANGELOG.md).


