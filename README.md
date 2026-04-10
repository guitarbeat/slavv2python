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
- **Acceleration**: Numba (optional)
- **Frameworks**: Streamlit (Web App), argparse (CLI)
- **Tooling**: Ruff (formatting/linting), MyPy (type checking), Pytest (testing)
- **Package Manager**: `pip` (with `setuptools`)

## Requirements

- **Python**: 3.9, 3.10, 3.11, or 3.12.
- **MATLAB** (Optional): Required only for parity comparisons and MATLAB result imports.

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

## Entry Points

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
from slavv import SLAVVProcessor

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

### Helper Scripts

- `.\make.ps1`: Windows PowerShell helper for common tasks.
  - `.\make.ps1 install`: Install the full contributor toolchain with `.[app,dev]`.
  - `.\make.ps1 format`: Format code using Ruff.
  - `.\make.ps1 lint`: Run non-mutating Ruff lint checks.
  - `.\make.ps1 typecheck`: Run MyPy.
  - `.\make.ps1 test`: Run all tests.
- `Makefile`: POSIX-style helper for Unix systems.

### Maintenance and Utility Scripts

Located in `workspace/scripts/`:
- `source/slavv/apps/parity_cli.py`: Canonical implementation of the parity comparison CLI.
- `workspace/scripts/cli/compare_matlab_python.py`: Backward-compatible wrapper for parity validation workflows.
- `workspace/scripts/maintenance/`: Scripts for repo mapping and MATLAB audit helpers.
- `workspace/scripts/benchmarks/`: Ad-hoc benchmark helpers that are not part of pytest collection.

## Testing

Run tests using `pytest`:

```powershell
# Run unit and integration tests
python -m pytest -m "unit or integration"

# Full suite including slow and UI tests
python -m pytest

# Diagnostic tests for environment/MATLAB parity setup
python -m pytest tests/diagnostic/test_comparison_setup.py
```

## Environment Variables

The application handles most environment settings internally, especially for Windows console support:
- `PYTHONUTF8`: Set to `1` by `slavv-app` to ensure UTF-8 support on Windows.
- `PYTHONIOENCODING`: Set to `utf-8` by `slavv-app`.

No manual environment variable configuration is typically required for standard use.

## Project Structure

| Path | Description |
| --- | --- |
| `source/slavv/` | Core package code (processing, I/O, analysis, visualization, CLI/app). |
| `tests/` | Unit, integration, UI, regression, and diagnostic tests. |
| `workspace/scripts/` | MATLAB comparison wrappers and maintenance helpers. |
| `workspace/reports/` | Archived tooling snapshots and repo-local reference artifacts. |
| `docs/` | Translation guide, MATLAB mapping, comparison layout references, and active chapter docs. |
| `external/` | Optional checkouts like `Vectorization-Public` (MATLAB SLAVV). |
| `data/` | Sample data and test volumes. |

## Documentation Path

For maintained project context, read in this order:

1. [Repository overview](README.md)
2. [Contributor workflow commands](AGENTS.md)
3. [MATLAB Translation Guide](docs/reference/MATLAB_TRANSLATION_GUIDE.md)
4. [MATLAB Mapping](docs/reference/MATLAB_MAPPING.md)
5. [Comparison Run Layout](docs/reference/COMPARISON_LAYOUT.md)
6. [Shared Candidate Generation Alignment](docs/chapters/shared-candidate-generation/README.md)

## License

This project is licensed under the **GNU GPL-3.0**. See the `LICENSE` file for details.

## TODOs / Known Issues

- [x] Complete full-package type hint coverage (currently focused on entry points).
- [ ] Expand documentation for custom energy computation methods.
- [ ] Optimize peak memory usage during Hessian eigenvalue computation.
- [x] Document advanced `slavv analyze` metrics.
- [ ] TODO: Add detailed contributor guide for adding new extraction algorithms.
