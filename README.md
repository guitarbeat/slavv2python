# SLAVV Python Translation

This repository hosts a Python and Streamlit based reimplementation of the SLAVV (Segmentation-Less, Automated, Vascular Vectorization) algorithm along with documentation of the original MATLAB project.

## Repository structure

- **slavv-streamlit/** – Streamlit application containing the Python port of the algorithm
- **Vectorization-Public/** – snapshot of the original MATLAB source code
- **slavv-matlab-docs/** – auto-generated reference documentation for the MATLAB code
- **docs/PORTING_SUMMARY.md** – combined improvements overview and source comparison
- **docs/MATLAB_TO_PYTHON_MAPPING.md** – MATLAB→Python function mapping with parity levels
- **docs/RESUME_INSTRUCTIONS.md** – notes on resuming work in a new environment
- **AGENTS.md** – contributor guidelines for automated and human maintainers

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
import numpy as np
from vectorization_core import SLAVVProcessor  # add slavv-streamlit/src to PYTHONPATH

volume = np.random.rand(16, 16, 16).astype(np.float32)
processor = SLAVVProcessor()
vertices, edges, network = processor.process_image(volume, voxel_size=(1, 1, 1))
print(len(vertices), len(edges))
```

## Testing

Verify that the environment is configured correctly by running the compile and test suite:

```bash
python -m py_compile $(git ls-files '*.py')
pytest -q
```

## Troubleshooting

- **ModuleNotFoundError: No module named 'src'** – ensure commands are executed from the `slavv-streamlit` directory or add it to `PYTHONPATH`.
- **ValueError: expected 3D TIFF** – `load_tiff_volume` only accepts grayscale, volumetric TIFFs.
- **High memory usage** – enable memory mapping with `load_tiff_volume(..., memory_map=True)` or reduce tile sizes via `max_voxels_per_node_energy`.

For additional details on algorithm features and usage see [slavv-streamlit/README.md](slavv-streamlit/README.md). The original MATLAB documentation is available in [slavv-matlab-docs/docs/index.md](slavv-matlab-docs/docs/index.md). See the MATLAB→Python parity mapping in [docs/MATLAB_TO_PYTHON_MAPPING.md](docs/MATLAB_TO_PYTHON_MAPPING.md).

## License

This project is distributed under the terms of the GNU GPL‑3.0 license, consistent with the upstream SLAVV repository.
