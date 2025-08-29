# SLAVV - Streamlit Implementation

This is a Python/Streamlit implementation of the SLAVV (Segmentation-Less, Automated, Vascular Vectorization) algorithm for analyzing vascular networks from grayscale, volumetric images.

## Original Repository

This implementation is based on the MATLAB code from:
https://github.com/UTFOIL/Vectorization-Public

## Features

- **Image Upload & Processing**: Upload TIFF images and process them with the SLAVV algorithm
- **Machine Learning Curation**: Use ML models to automatically curate detected vertices and edges
- **Interactive Visualization**: View results in 2D and 3D with various coloring schemes
- **Export Capabilities**: Export results in multiple formats (VMV, CASX, MAT, CSV, JSON)
- **Statistics Dashboard**: View detailed network statistics and histograms

## Installation

1. Clone this repository

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to the displayed URL (typically http://localhost:8501)

3. Upload a TIFF image of vascular structure

4. Configure processing parameters:
   - Voxel size in microns
   - Vessel radius range
   - Microscopy parameters

5. Start vectorization and wait for processing to complete

6. Use the ML Curation page to refine results

7. Visualize the network in 2D or 3D

8. Export results in your preferred format

## Algorithm Overview

The SLAVV algorithm works in four main steps:

1. **Energy Image Formation**: Multi-scale Hessian-based filtering to enhance vessel centerlines
2. **Vertex Extraction**: Detection of local minima in the energy image as vessel points
3. **Edge Extraction**: Tracing paths between vertices following energy gradients
4. **Network Extraction**: Building the final connected network structure

## Implementation Notes

This Python implementation provides:

- Core vectorization functionality translated from MATLAB
- Simplified but functional versions of the main algorithms
- Integration with scikit-learn for curation workflows
- Interactive web interface using Streamlit
- Visualization using Plotly for 2D/3D plots
For port status and parity notes, see [../docs/MATLAB_TO_PYTHON_MAPPING.md](../docs/MATLAB_TO_PYTHON_MAPPING.md).

## Limitations

This is a simplified implementation of the full MATLAB version. Some advanced features may not be fully implemented:

- Complete PSF modeling
- All original visualization options
- Full parameter optimization
- Complete file format support

## Programmatic usage

If you want to import the core module directly in Python, add the source directory to `PYTHONPATH`:

```bash
export PYTHONPATH=slavv-streamlit/src
python - << 'PY'
from vectorization_core import SLAVVProcessor
print(SLAVVProcessor)
PY
```

## Citation

If you use this software, please cite the original SLAVV methodology paper:

```
@article{mihelic2021segmentation,
  title={Segmentation-Less, Automated, Vascular Vectorization},
  author={Mihelic, Samuel A and Sikora, William A and Hassan, Ahmed M and Williamson, Michael R and Jones, Theresa A and Dunn, Andrew K},
  journal={PLOS Computational Biology},
  volume={17},
  number={10},
  pages={e1009451},
  year={2021},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

## License

This implementation follows the same GNU GPL-3.0 license as the original repository.
