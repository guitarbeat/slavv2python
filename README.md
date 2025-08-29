# SLAVV Python Translation

This repository hosts a Python and Streamlit based reimplementation of the SLAVV (Segmentation-Less, Automated, Vascular Vectorization) algorithm along with documentation of the original MATLAB project.

## Repository structure

- **slavv-streamlit/** – Streamlit application containing the Python port of the algorithm
- **Vectorization-Public/** – snapshot of the original MATLAB source code
- **slavv-matlab-docs/** – auto-generated reference documentation for the MATLAB code
- **docs/index.md** – documentation index (User, Developer, Mapping, Coverage)
- **docs/PORTING_SUMMARY.md** – combined improvements overview and source comparison
- **docs/MATLAB_TO_PYTHON_MAPPING.md** – MATLAB→Python function mapping with parity levels
- **docs/MATLAB_COVERAGE_REPORT.md** – inventory of MATLAB files and coverage status
- **docs/TESTING.md** – how to run compile checks and tests
- **CONTRIBUTING.md** – contributor guidelines for automated and human maintainers

## Getting started

1. Clone this repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   cd slavv-streamlit
   pip install -r requirements.txt
   ```
4. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. Open the provided URL in your browser and follow the on-screen instructions to process images.

## Usage

Run the pipeline programmatically with `SLAVVProcessor`:

```python
import sys, numpy as np

# Ensure the source path is importable
sys.path.append('slavv-streamlit/src')  # or: export PYTHONPATH=slavv-streamlit/src

from vectorization_core import SLAVVProcessor

volume = np.random.rand(16, 16, 16).astype(np.float32)
params = {"microns_per_voxel": [1.0, 1.0, 1.0]}
processor = SLAVVProcessor()
results = processor.process_image(volume, params)

print(len(results['vertices']['positions']), len(results['edges']['traces']))
```

## Testing

Verify that the environment is configured correctly by running the compile and test suite:

```bash
python -m py_compile $(git ls-files '*.py')
pytest -q
```

## Troubleshooting

- **ImportError for `vectorization_core`** – ensure `slavv-streamlit/src` is on `PYTHONPATH` (see example above).
- **ValueError: expected 3D TIFF** – `load_tiff_volume` only accepts grayscale, volumetric TIFFs.
- **High memory usage** – enable memory mapping with `load_tiff_volume(..., memory_map=True)` or reduce tile sizes via `max_voxels_per_node_energy`.

For additional details on algorithm features and usage see [slavv-streamlit/README.md](slavv-streamlit/README.md). The original MATLAB documentation is available in [slavv-matlab-docs/docs/index.md](slavv-matlab-docs/docs/index.md). See the MATLAB→Python parity mapping in [docs/MATLAB_TO_PYTHON_MAPPING.md](docs/MATLAB_TO_PYTHON_MAPPING.md) and the coverage report in [docs/MATLAB_COVERAGE_REPORT.md](docs/MATLAB_COVERAGE_REPORT.md).

## License

This project is distributed under the terms of the GNU GPL‑3.0 license, consistent with the upstream SLAVV repository.
