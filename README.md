# SLAVV Python Port

Python and Streamlit reimplementation of **SLAVV** (Segmentation-Less, Automated, Vascular Vectorization) for 3D vascular network vectorization from microscopy volumes. This repository includes the core library, a web UI, and tooling to compare results with the original MATLAB implementation.

## Repository structure

| Path | Description |
|------|-------------|
| **src/slavv/** | Core Python package (energy, tracing, graph, I/O, visualization) |
| **src/slavv/apps/** | Web applications (`web_app.py`) |
| **examples/** | Example scripts (e.g. `run_tutorial.py`) |
| **scripts/** | Developer tools for MATLAB comparison and validation |
| **tests/** | Unit, integration, and UI tests |
| **docs/** | [Documentation index](docs/README.md): architecture, development, migration |
| **legacy/** | Original MATLAB source (`Vectorization-Public`) and scripts |
| **external/** | Large binary dependencies (e.g. `blender_resources`) |
| **CONTRIBUTING.md** | Contribution guidelines and testing |
| **pyproject.toml** | Package metadata and dependencies |

## Getting started

1. Clone this repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```
3. Install the package in editable mode (installs dependencies):
   ```bash
   pip install -e .
   ```
4. Launch the Streamlit application:
   ```bash
   streamlit run src/slavv/apps/web_app.py
   ```
5. Open the provided URL in your browser.

## Usage

### Programmatic Usage (Headless/Batch)
For integration into other pipelines or running on a cluster without the UI, use the `SLAVVProcessor` class.
See **`examples/run_tutorial.py`** or the docstrings for details.

```python
from slavv import SLAVVProcessor

# Initialize
processor = SLAVVProcessor()

# Run
results = processor.process_image(image_data, params)

print(f"Vertices: {len(results['vertices']['positions'])}")
print(f"Edges: {len(results['edges']['traces'])}")
```



## Testing

Verify that the environment is configured correctly by running the test suite:

```bash
python -m pytest tests/
```

## Troubleshooting

- **ImportError for `src.slavv`** – Ensure you are running Python from the repository root.
- **ValueError: expected 3D TIFF** – `load_tiff_volume` only accepts grayscale, volumetric TIFFs.
- **High memory usage** – enable memory mapping with `load_tiff_volume(..., memory_map=True)` or reduce tile sizes via `max_voxels_per_node_energy`.

For the canonical port status, see [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md).

## License

This project is licensed under the [GNU GPL-3.0](LICENSE) license, consistent with the upstream SLAVV (Vectorization-Public) repository.
