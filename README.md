# SLAVV Python Port

This repository hosts a Python and Streamlit based reimplementation of the SLAVV (Segmentation-Less, Automated, Vascular Vectorization) algorithm along with documentation of the original MATLAB project.

## Repository structure

- **src/slavv/** – Main Python package containing the algorithm core, I/O, and visualization tools
- **app.py** – Streamlit application entry point
- **Vectorization-Public/** – snapshot of the original MATLAB source code
- **docs/ARCHITECTURE.md** – System architecture and data flow diagrams
- **docs/DEVELOPMENT.md** – Status, features, performance, roadmap
- **docs/MIGRATION_GUIDE.md** – Canonical MATLAB→Python mapping
- **CONTRIBUTING.md** – Guidelines, coding standards, and testing commands
- **tests/** – Comprehensive test suite

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
   streamlit run app.py
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

This project is distributed under the terms of the GNU GPL‑3.0 license, consistent with the upstream SLAVV repository.
