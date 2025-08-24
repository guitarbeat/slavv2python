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
    - [x] `calculate_energy_field` parity with `get_energy_V202.m`
      - [x] Verify scale schedule: `scales_per_octave`, ordination, and min-projection across scales
      - [x] Match PSF handling and filter ratios (`gaussian_to_ideal_ratio`, `spherical_to_annular_ratio`)
      - [x] Account for anisotropic voxels in smoothing and Hessian calculations
      - [x] Align vesselness/energy sign convention and threshold semantics
      - [x] Validate pixel↔micron conversions for radii and PSF sigmas
  - [x] `extract_vertices` parity with `get_vertices_V200.m`
    - [x] Local minima detection structuring element matches MATLAB behavior
    - [x] `energy_upper_bound`, `space_strel_apothem`, `length_dilation_ratio` semantics
    - [x] Volume-exclusion logic parity (ordering, tie-breaking, distance metric)
    - [x] `extract_edges` parity with `get_edges_V300.m`
    - [x] Implement proper gradient-descent ridge following (use energy gradients)
    - [x] Step size per origin radius and adaptive stepping termination

    - [x] Terminal detection: near-vertex, energy rise, out-of-bounds, max steps
    - [x] Implement/restore `_find_terminal_vertex` or remove call if redundant with `_near_vertex`
    - [x] Avoid duplicate/self edges; limit `number_of_edges_per_vertex`
  - [x] `construct_network` parity with `get_network_V190.m`
    - [x] Adjacency construction and symmetric connectivity
    - [x] Strand/connected component tracing and bifurcation detection
    - [x] Deduplicate edges; stable edge keying; retain edge traces
  - [x] Helper parity and units
    - [x] `_near_vertex` uses correct radius units (voxel vs micron); consistent with radii arrays
    - [x] `_compute_gradient` handles anisotropic voxels; central differences validated
    - [x] `_in_bounds` checks consistent with array indexing order

  - [ ] I/O and outputs
    - [x] Confirm returned structures, dtypes, and shapes match expected consumers
  - [x] Document and test public API for stability
  - [ ] Parity validation
    - [ ] Add small-volume regression comparisons vs MATLAB outputs
    - [ ] Record deviations and rationale where exact parity is not feasible
- [x] Fix indentation and duplicate helper definitions in `src/vectorization_core.py` (syntax error around line ~443)
- [x] Pass `vertex_scales` into `_trace_edge`; make `_near_vertex` return index; deduplicate `_trace_strand`/`_in_bounds`
- [x] Standardize `lumen_radius_pixels` to scalar per-scale; correct Hessian `sigma` usage (scalar)

- [x] Improve energy field to more closely match `get_energy_V202.m` filter kernels and PSF weighting
- [x] Optimize vertex volume exclusion and geometry parity with `get_vertices_V200.m`
- [x] Implement proper gradient-descent ridge following for edges (closer to `get_edges_V300.m`)
- [x] Add network cleaning steps (hairs/orphans/cycles) per MATLAB scripts
  - [x] Remove short hairs and track orphan vertices in `construct_network`
  - [x] Prune cycles during network construction
- [x] Add preprocessing parity (intensity normalization, band fixes) inspired by `pre_processing.m` and `fix_intensity_bands.m`
- [x] Implement chunked/tiling processing using `get_chunking_lattice_V190.m` honoring `max_voxels_per_node_energy`
- [x] Implement vessel direction estimation parity (`get_vessel_directions_V2/V3/V5.m`) for better initial edge directions
- [x] Add watershed-based edge alternative (`get_edges_by_watershed.m`) as a selectable method
  - [x] Implement strand combining/sorting/mismatch fixes (`combine_strands.m`, `sort_network_V180.m`, `fix_strand_vertex_mismatch*.m`)


## 2. ML Curation

- [ ] Integrate actual machine learning models for vertex and edge curation in `src/ml_curator.py`.
- [ ] Provide options for users to upload their own pre-trained models or training data.
- [ ] Develop a clear workflow for training and deploying ML models within the Streamlit app.
- [ ] Add training workflow and dataset ingestion in app; persist/load models
- [ ] Port feature extraction parity from `vertex_feature_extractor.m` and `edge_info_extractor.m` (feature set alignment)
- [ ] Align curator network behavior with `vertexCuratorNetwork_V*` and `edgeCuratorNetwork_V*`
- [ ] Implement `choose_vertices_V200.m` and `choose_edges_V200.m` logic as additional heuristics
- [ ] Add uncurated info extraction parity (`uncuratedInfoExtractor.m`) for QA datasets

## 3. Visualization Enhancements

- [ ] Refine 2D and 3D network visualizations for better clarity and interactivity.
- [ ] Add more visualization options (e.g., coloring by other metrics, interactive slicing for 3D data).
- [x] Implement visualization for the energy field with selectable axis and slice index.
- [ ] Color edges by average energy/strand id with legends; improve 3D color/opacity mapping
- [ ] Add depth coloring parity (`visualize_depth_via_color_V200.m`)
- [ ] Add strand coloring in 3D parity (`visualize_strands_via_color_3D_V2/V3.m`)
- [ ] Add animated strand visualization parity (`animate_strands_3D.m`)
- [ ] Explore flow field rendering parity (`render_flow_field_V3/V4.m`)

## 4. Error Handling and Robustness

- [ ] Improve error handling for file uploads (e.g., non-TIFF files, corrupted TIFFs).
- [x] Fix statistics page to use available data and avoid KeyErrors; guard metrics with defaults
- [ ] Enhance parameter validation with more specific error messages and suggestions.
- [ ] Implement robust handling for edge cases in image processing and network construction.
- [x] Implement cropping helpers parity (`crop_vertices_V200.m`, `crop_edges_V200.m`, `crop_vertices_by_mask.m`)

## 5. Performance Optimization

- [ ] Profile the core SLAVV algorithm functions to identify performance bottlenecks.
- [ ] Explore using Numba or Cython for computationally intensive parts of the code.
- [ ] Optimize data structures and algorithms for better memory usage and speed.
- [x] Implement chunking lattice and on-the-fly tiling (parity with `get_chunking_lattice_V190.m`)
- [ ] Evaluate memory mapping/HDF5 streaming for large volumes (parity with MATLAB I/O patterns)
- [ ] Consider scikit-image Sato/Frangi optimized paths or custom vectorized kernels

## 6. Testing

- [ ] Develop comprehensive unit tests for all functions in `src/vectorization_core.py`, `src/ml_curator.py`, and `src/visualization.py`.
- [ ] Add regression tests comparing against small MATLAB-ground-truth outputs where feasible
- [ ] Implement integration tests for the Streamlit application to ensure end-to-end functionality.
- [ ] Set up a continuous integration (CI) pipeline for automated testing.
- [x] Add basic tests for network statistics helper (`calculate_network_statistics.m`)
- [ ] Add parity tests for surface area (`calculate_surface_area.m`)

## 7. Documentation

- [ ] Add detailed docstrings to all functions and classes in the codebase.
- [ ] Expand the `README.md` with more detailed setup instructions, usage examples, and troubleshooting tips.
- [x] Create a dedicated `docs` directory for comprehensive user and developer documentation.
- [x] Add a MATLAB→Python function mapping table with parity level (exact/approx/omitted) — see `docs/MATLAB_TO_PYTHON_MAPPING.md`.

## 8. User Experience (UX) Enhancements

- [ ] Improve the responsiveness and layout of the Streamlit application for different screen sizes.
- [ ] Add tooltips and help texts for all input parameters and metrics.
- [ ] Implement a progress tracker for long-running operations.
- [ ] Provide clear feedback to the user at each step of the processing pipeline.

## 9. Export Formats

- [x] Ensure full support for all export formats mentioned in the original MATLAB SLAVV documentation (VMV, CASX, MAT, CSV, JSON).
- [x] Complete VMV/CASX specs; add MAT export
- [ ] Add import support for CASX/VMV/MAT (parity with `casx_mat2file.m`, `casX2mat.m`, `vmv_mat2file.m`, `strand2vmv.m`, `strand2casx.m`)

## Notes

- Progress items were verified against original MATLAB references (e.g., `get_energy_V202.m`, `get_vertices_V200.m`, `get_edges_V300.m`) where applicable.