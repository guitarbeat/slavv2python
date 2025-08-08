# SLAVV2Python - To-Do List

This document outlines planned improvements and added features for the SLAVV2Python project.

## 1. Core SLAVV Algorithm Implementation

- [x] Complete the implementation of `extract_vertices` in `src/vectorization_core.py`.
- [x] Complete the implementation of `_generate_edge_directions` in `src/vectorization_core.py`.
- [x] Complete the implementation of `_trace_edge` in `src/vectorization_core.py`.
- [x] Complete the implementation of `_find_terminal_vertex` in `src/vectorization_core.py`.
- [x] Complete the implementation of `_near_vertex` in `src/vectorization_core.py`.
- [x] Complete the implementation of `extract_edges` in `src/vectorization_core.py`.
- [x] Complete the implementation of `construct_network` in `src/vectorization_core.py`.
- [ ] Ensure all core functions in `src/vectorization_core.py` accurately reflect the MATLAB algorithm.
- [x] Fix indentation and duplicate helper definitions in `src/vectorization_core.py` (syntax error around line ~443)
- [x] Pass `vertex_scales` into `_trace_edge`; make `_near_vertex` return index; deduplicate `_trace_strand`/`_in_bounds`
- [x] Standardize `lumen_radius_pixels` to scalar per-scale; correct Hessian `sigma` usage (scalar)
- [ ] Improve energy field to more closely match `get_energy_V202.m` filter kernels and PSF weighting
- [ ] Optimize vertex volume exclusion and geometry parity with `get_vertices_V200.m`
- [ ] Implement proper gradient-descent ridge following for edges (closer to `get_edges_V300.m`)
- [ ] Add network cleaning steps (hairs/orphans/cycles) per MATLAB scripts

## 2. ML Curation

- [ ] Integrate actual machine learning models for vertex and edge curation in `src/ml_curator.py`.
- [ ] Provide options for users to upload their own pre-trained models or training data.
- [ ] Develop a clear workflow for training and deploying ML models within the Streamlit app.
- [ ] Add training workflow and dataset ingestion in app; persist/load models

## 3. Visualization Enhancements

- [ ] Refine 2D and 3D network visualizations for better clarity and interactivity.
- [ ] Add more visualization options (e.g., coloring by other metrics, interactive slicing for 3D data).
- [x] Implement visualization for the energy field with selectable axis and slice index.
- [ ] Color edges by average energy/strand id with legends; improve 3D color/opacity mapping

## 4. Error Handling and Robustness

- [ ] Improve error handling for file uploads (e.g., non-TIFF files, corrupted TIFFs).
- [x] Fix statistics page to use available data and avoid KeyErrors; guard metrics with defaults
- [ ] Enhance parameter validation with more specific error messages and suggestions.
- [ ] Implement robust handling for edge cases in image processing and network construction.

## 5. Performance Optimization

- [ ] Profile the core SLAVV algorithm functions to identify performance bottlenecks.
- [ ] Explore using Numba or Cython for computationally intensive parts of the code.
- [ ] Optimize data structures and algorithms for better memory usage and speed.

## 6. Testing

- [ ] Develop comprehensive unit tests for all functions in `src/vectorization_core.py`, `src/ml_curator.py`, and `src/visualization.py`.
- [ ] Add regression tests comparing against small MATLAB-ground-truth outputs where feasible
- [ ] Implement integration tests for the Streamlit application to ensure end-to-end functionality.
- [ ] Set up a continuous integration (CI) pipeline for automated testing.

## 7. Documentation

- [ ] Add detailed docstrings to all functions and classes in the codebase.
- [ ] Expand the `README.md` with more detailed setup instructions, usage examples, and troubleshooting tips.
- [ ] Create a dedicated `docs` directory for comprehensive user and developer documentation.

## 8. User Experience (UX) Enhancements

- [ ] Improve the responsiveness and layout of the Streamlit application for different screen sizes.
- [ ] Add tooltips and help texts for all input parameters and metrics.
- [ ] Implement a progress tracker for long-running operations.
- [ ] Provide clear feedback to the user at each step of the processing pipeline.

## 9. Export Formats

- [ ] Ensure full support for all export formats mentioned in the original MATLAB SLAVV documentation (VMV, CASX, MAT, CSV, JSON).
- [ ] Complete VMV/CASX specs; add MAT export

## Notes

- Progress items were verified against original MATLAB references (e.g., `get_energy_V202.m`, `get_vertices_V200.m`, `get_edges_V300.m`) where applicable.



