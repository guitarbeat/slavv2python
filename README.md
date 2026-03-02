# SLAVV Python Port

Python and Streamlit reimplementation of **SLAVV** (Segmentation-Less, Automated, Vascular Vectorization) for 3D vascular network vectorization from microscopy volumes. This repository includes the core library, a web UI, and tooling to compare results with the original MATLAB implementation.

## Repository structure

| Path | Description |
|------|-------------|
| **source/slavv/** | Core Python package (energy, tracing, graph, I/O, visualization) |
| **source/slavv/apps/** | Web applications (`web_app.py`) |
| **workspace/scripts/** | Setup, CLI wrappers, and MATLAB integration |
| **workspace/examples/** | Programmatic usage examples (`run_tutorial.py`) |
| **workspace/notebooks/** | Interactive Jupyter workflows and comparison dashboards |
| **workspace/experiments/** | Output directory for runs and comparisons |
| **tests/** | Unit, integration, and UI tests |
| **docs/** | Development, archive summary, and workspace reference docs |
| **external/Vectorization-Public/** | Original MATLAB source |
| **external/** | Large binary dependencies (e.g., `blender_resources`) |
| **pyproject.toml** | Package metadata and dependencies |

## Getting started

### Quick Setup (Windows)

Run the automated setup script:
```powershell
.\workspace\scripts\setup\setup_env.ps1
```

This will:
- Create a virtual environment (venv or conda)
- Install all dependencies from `pyproject.toml`
- Register a Jupyter kernel for notebooks
- Guide you through the next steps

### Manual Setup

1. **Clone this repository**:
   ```bash
   git clone https://github.com/UTFOIL/slavv2python.git
   cd slavv2python
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Option A: venv (built-in)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows
   source .venv/bin/activate      # Linux/Mac

   # Option B: conda
   conda create -n slavv-env python=3.10
   conda activate slavv-env
   ```

3. **Install the package in editable mode**:
   ```bash
   pip install -e .
   ```
   This installs all dependencies from `pyproject.toml` including matplotlib, numpy, scipy, etc.

4. **For notebook usage, install Jupyter**:
   ```bash
   pip install jupyter ipykernel
   python -m ipykernel install --user --name=slavv-env --display-name="Python (SLAVV)"
   ```

5. **Validate your setup**:
   ```bash
   jupyter notebook workspace/notebooks/00_Setup_and_Validation.ipynb
   # Select kernel: "Python (SLAVV)"
   ```

### Launch the Web Application

```bash
streamlit run source/slavv/apps/web_app.py
# Or use the entry point: slavv-app
```
Open the provided URL in your browser.

## Usage

### Programmatic Usage (Headless/Batch)
For integration into other pipelines or running on a cluster without the UI, use the `SLAVVProcessor` class.
See **`workspace/examples/run_tutorial.py`** or the docstrings for details.

```python
from slavv import SLAVVProcessor

# Initialize
processor = SLAVVProcessor()

# Run
results = processor.process_image(image_data, params)

print(f"Vertices: {len(results['vertices']['positions'])}")
print(f"Edges: {len(results['edges']['traces'])}")
```

### Visualization and export

SLAVV exports `VMV`, `CASX`, `CSV`, and `JSON`.

- `VMV` is intended for Blender + VessMorphoVis.
- `CASX` is XML-based vascular network data.
- `CSV` and `JSON` are convenient for analysis pipelines.

To inspect outputs in Blender:
1. Install Blender and the VessMorphoVis add-on.
2. Run SLAVV and generate a `.vmv` file.
3. In Blender, load the `.vmv` file from the VessMorphoVis panel.

For experiment-level inspection, use `workspace/notebooks/04_Comparison_Dashboard.ipynb`.

## Testing

Verify that the environment is configured correctly by running the test suite:

```bash
python -m pytest tests/
```

Fast lane (used by CI for quick feedback):

```bash
python -m pytest -m "unit or integration"
```

## Contributing

### Code and docs
- Place package code under `source/slavv/`.
- Place tests under `tests/` by category (`unit`, `integration`, `ui`, `diagnostic`).
- Place supplementary docs under `docs/`.
- Use relative links between repo documents.

### Style
- Follow PEP 8.
- Public functions in `source/` should use type hints.
- Use docstrings for exported members.
- Use `logging` in library code (avoid `print()` in `source/`).

### Checks before opening a PR
```bash
python -m compileall source/ tests/
python -m pytest -m "unit or integration"
python -m pytest tests/ -v
```

### Regression guardrails
- Do not break existing tests.
- Preserve MATLAB parity for core algorithms.
- Prefer behavior-level tests with deterministic fixtures.
- For float expectations, use `np.allclose(..., atol=1e-6)`.

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for testing and repository structure policy.
See [docs/WORKSPACE.md](docs/WORKSPACE.md) for workspace scripts/notebooks/experiments conventions.

## Troubleshooting

- **ImportError for `slavv`** - Ensure you are running Python from the repository root and have run `pip install -e .`.
- **ValueError: expected 3D TIFF** - `load_tiff_volume` only accepts grayscale, volumetric TIFFs.
- **High memory usage** - enable memory mapping with `load_tiff_volume(..., memory_map=True)` or reduce tile sizes via `max_voxels_per_node_energy`.
- **Wrong Jupyter kernel** - run `python -m ipykernel install --user --name=slavv-env --display-name=\"Python (SLAVV)\"` and select that kernel.

For migration and parity status, see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## License

This project is licensed under the [GNU GPL-3.0](LICENSE) license, consistent with the upstream SLAVV (Vectorization-Public) repository.
