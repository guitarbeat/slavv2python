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
The core `SLAVVProcessor` acts as an agnostic API backend for the entire pipeline. It is entirely decoupled from any user interface (like Streamlit or the CLI), meaning you can easily import it into your own custom Python scripts, Jupyter notebooks, or wrap it in a REST API (like FastAPI) to process remote jobs.

For integration into other pipelines or running on a cluster without the UI, use the `SLAVVProcessor` class.
See **`workspace/examples/run_tutorial.py`** or the docstrings for details.

```python
from slavv import SLAVVProcessor

# Initialize the agnostic backend processor
processor = SLAVVProcessor()

# Run the pipeline (takes pure numpy arrays and a parameter dict)
results = processor.process_image(image_data, params)

# Returns a pure Python dictionary of vectorization features
print(f"Vertices: {len(results['vertices']['positions'])}")
print(f"Edges: {len(results['edges']['traces'])}")
```

### Checkpointing and Manual Curation (Resuming)

Just like the original MATLAB implementation's ability to pick up in the middle of a workflow (e.g., after manual curation of vertices or edges), the Python `SLAVVProcessor` supports intermediate checkpointing. 

By passing a `checkpoint_dir` to `process_image()`, the pipeline will automatically save intermediate steps to disk (`checkpoint_energy.pkl`, `checkpoint_vertices.pkl`, `checkpoint_edges.pkl`, and `checkpoint_network.pkl`). 

If you run the pipeline again with the same `checkpoint_dir`, it will automatically detect these files, skip the expensive computations for those prior steps, and load the cached (or manually curated) data instead. This enables seamless resumption of crashed runs, iterative parameter tuning on downstream steps, and injecting manual external curation natively.

```python
from slavv import SLAVVProcessor

processor = SLAVVProcessor()

# Run the pipeline with checkpointing enabled.
# If 'my_run_checkpoints/checkpoint_vertices.pkl' already exists (e.g., from a previous run
# or after an external script manually curated the vertices), the pipeline will automatically 
# load it, skip the Energy and Vertex Extraction steps, and resume directly at Edge Extraction!
results = processor.process_image(
    image_data, 
    params, 
    checkpoint_dir="my_run_checkpoints"
)
```

### Interactive Curation (Graphical Curator Interface)

Just like the original MATLAB GCI, the Python port provides a **4-panel desktop application** for manual curation of vertices and edges. The interface includes:

- **Volume Map** – 3D bounding box showing your current field of view
- **Volume Display** – 2D MIP with Depth/Thickness sliders and X/Y/Z orthogonal views
- **Intensity Histogram** – pixel distribution with brightness/contrast controls
- **Energy Histogram** – draggable threshold line for global energy thresholding

Curation actions: **Toggle** individual vertices/edges between True (blue) and False (red), **Sweep** to hide red objects, and **Add** edges by clicking two vertices.

Launch from the Streamlit app (**Curation → Interactive (Manual GUI) → Launch**) or programmatically:

```python
from slavv.analysis.interactive_curator import run_curator

curated_vertices, curated_edges = run_curator(energy_data, vertices, edges)
```

### MATLAB Cross-Compatibility

You can **import curated data from MATLAB** and continue the pipeline in Python. This enables workflows like: run Energy + Vertex Extraction in MATLAB → curate vertices in MATLAB's GCI → import into Python → run Edge Extraction and Network in Python.

**Via CLI:**
```bash
# Import a MATLAB batch folder as Python checkpoints
slavv import-matlab -b path/to/batch_260210-101213 -c my_checkpoints/

# Resume the pipeline in Python (skips steps that have checkpoints)
slavv run -i volume.tif --checkpoint-dir my_checkpoints/ --export csv json
```

**Via Python:**
```python
from slavv.io.matlab_bridge import import_matlab_batch
from slavv import SLAVVProcessor

# Convert MATLAB batch output → Python checkpoint pickles
import_matlab_batch("path/to/batch_260210-101213", "my_checkpoints/")

# Pipeline auto-detects checkpoints and skips those stages
processor = SLAVVProcessor()
results = processor.process_image(image, params, checkpoint_dir="my_checkpoints/")
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
