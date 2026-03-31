# SLAVV Python Port

Python reimplementation of **SLAVV** (Segmentation-Less, Automated, Vascular Vectorization) for 3D vascular network extraction from microscopy volumes. This repository contains the core package code, a Streamlit web application, a Command-Line Interface (CLI), MATLAB import helpers, and parity/comparison tooling used to validate the port.

## Overview

SLAVV provides an automated pipeline to extract vascular graphs (vertices and edges) directly from 3D image data without an intermediate segmentation step. This Python port aims for high parity with the original MATLAB implementation while providing a modern, scalable, and easy-to-use Python interface.

## Requirements

- **Python**: 3.9, 3.10, 3.11, or 3.12.
- **Key Dependencies**: NumPy, SciPy, scikit-image, scikit-learn, NetworkX, h5py, Tifffile, Matplotlib, Plotly, Pandas.
- **Optional**: Streamlit (for the web app), Numba (for acceleration).

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
   ```

3. **Install pre-commit hooks** (recommended for contributors):

   ```powershell
   pre-commit install
   ```

## Usage

### Command-Line Interface (CLI)

Use the `slavv` command for headless or scripted workflows:

```powershell
# Get system and version information
slavv info

# Run the full pipeline on a TIFF volume
slavv run -i volume.tif -o slavv_output --export csv json

# Run with custom parameters and resume support
slavv run -i volume.tif -o slavv_output --checkpoint-dir checkpoints --vessel-radius 2.0

# Import MATLAB results for comparison
slavv import-matlab -b path\to\matlab_batch -c my_checkpoints

# Analyze and plot results
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

### Programmatic Usage

```python
from slavv.core.pipeline import SLAVVProcessor

# Initialize the processor
processor = SLAVVProcessor()

# Run the pipeline
results = processor.process_image(
    image_data, 
    params={"vessel_radius": 1.5}, 
    event_callback=lambda e: print(f"[{e.stage}] {e.detail}")
)

print(f"Vertices: {len(results['vertices']['positions'])}")
print(f"Edges: {len(results['edges']['traces'])}")
```

## Scripts and Maintenance

Windows shortcut commands are available through `.\make.ps1`:

- `.\make.ps1 install`: Install dependencies.
- `.\make.ps1 format`: Format code using Ruff.
- `.\make.ps1 lint`: Lint code using Ruff.
- `.\make.ps1 test`: Run all tests.

For Unix-like systems, a `Makefile` is also provided.

## Project Structure

| Path | Description |
| --- | --- |
| `source/slavv/` | Core package code (processing, I/O, analysis, visualization, CLI/app). |
| `tests/` | Unit, integration, UI, regression, and diagnostic tests. |
| `workspace/scripts/` | MATLAB comparison wrappers and maintenance helpers. |
| `workspace/reports/` | Archived tooling snapshots and repo-local reference artifacts. |
| `docs/` | Documentation for MATLAB mapping, comparison layouts, and parity status. |
| `external/` | Optional checkouts like `Vectorization-Public` (MATLAB SLAVV). |
| `data/` | Sample data and test volumes. |

## MATLAB Parity and Comparisons

A key goal of this project is parity with the original MATLAB SLAVV implementation.

1. **Verify setup**: `python -m pytest tests/diagnostic/test_comparison_setup.py`
2. **Run comparison**:
   ```powershell
   python workspace/scripts/cli/compare_matlab_python.py `
       --input data/slavv_test_volume.tif `
       --matlab-path "C:\Path\To\MATLAB\bin\matlab.exe" `
       --output-dir comparison_output
   ```

Detailed parity findings and mapping notes are available in the `docs/` directory.

## Testing

Run tests using `pytest`:

```powershell
# Run unit and integration tests
python -m pytest -m "unit or integration"

# Full suite including slow and UI tests
python -m pytest
```

## Environment Variables

The application handles most environment settings internally (e.g., `PYTHONUTF8` for Windows console support). No manual environment variable configuration is typically required for standard use.

## License

This project is licensed under the **GNU GPL-3.0**. See the `LICENSE` file for details.

## TODOs / Known Issues

- [ ] Complete full-package type hint coverage (currently focused on entry points).
- [ ] Expand documentation for custom energy computation methods.
- [ ] Optimize peak memory usage during Hessian eigenvalue computation.
