# SLAVV Python Port

This repository hosts a Python and Streamlit based reimplementation of the SLAVV (Segmentation-Less, Automated, Vascular Vectorization) algorithm along with documentation of the original MATLAB project.

## Repository structure

- **src/slavv/** – Main Python package containing the algorithm core, I/O, and visualization tools
- **app.py** – Streamlit application entry point
- **Vectorization-Public/** – snapshot of the original MATLAB source code
- **docs/MATLAB_TO_PYTHON_MAPPING.md** – canonical MATLAB→Python mapping (includes coverage + deviations)
- **CONTRIBUTING.md** – contributing and testing guidelines
- **tests/** – Comprehensive test suite and benchmarks

## Getting started

1. Clone this repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. Open the provided URL in your browser and follow the instructions to process images.

## Usage

### Programmatic Usage (Headless/Batch)
For integration into other pipelines or running on a cluster without the UI, use the `SLAVVProcessor` class.
See **`examples/run_headless_demo.py`** for a complete example.

```python
from slavv.vectorization_core import SLAVVProcessor

# Initialize
processor = SLAVVProcessor()

# Run
results = processor.process_image(image_data, params)
```

### Tutorials
See **`examples/run_tutorial.py`** for a script that reproduces the steps from the original MATLAB tutorial (requires external data).
print(f"Vertices: {len(results['vertices']['positions'])}")
print(f"Edges: {len(results['edges']['traces'])}")
```

## Testing

Verify that the environment is configured correctly by running the test suite:

```bash
pytest
```

## Troubleshooting

- **ImportError for `src.slavv`** – Ensure you are running Python from the repository root.
- **ValueError: expected 3D TIFF** – `load_tiff_volume` only accepts grayscale, volumetric TIFFs.
- **High memory usage** – enable memory mapping with `load_tiff_volume(..., memory_map=True)` or reduce tile sizes via `max_voxels_per_node_energy`.

For the canonical port status, see [docs/MATLAB_TO_PYTHON_MAPPING.md](docs/MATLAB_TO_PYTHON_MAPPING.md).

## License

This project is distributed under the terms of the GNU GPL‑3.0 license, consistent with the upstream SLAVV repository.
