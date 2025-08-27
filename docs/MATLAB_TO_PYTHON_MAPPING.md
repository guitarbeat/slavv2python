## MATLAB → Python Mapping (Parity Levels)

This document maps key MATLAB SLAVV functions/scripts to their Python counterparts in this repository, and indicates the current parity level:
- Exact: Equivalent behavior intended and validated
- Approximate: Implemented and functional but not numerically identical or missing some heuristics/options
- Omitted: Not implemented yet (planned)

See [PARITY_DEVIATIONS.md](PARITY_DEVIATIONS.md) for rationale behind notable differences.

### Core Algorithm

| MATLAB | Python | Parity | Notes |
|---|---|---|---|
| `vectorize_V200.m` | `slavv-streamlit/src/vectorization_core.py:process_image` | Approximate | Validates 3D inputs and gracefully handles empty vertices/edges. |
| `get_energy_V202.m` | `slavv-streamlit/src/vectorization_core.py:calculate_energy_field` | Approximate | Multi-scale Hessian energy with per-scale PSF-weighted smoothing, configurable Gaussian/annular ratios, sign-aware vesselness, and validated pixel↔micron conversions for radii/PSF; scale schedule and anisotropic voxels verified though kernel details remain simplified. Defaults avoid storing all per-scale volumes unless `return_all_scales=True` to reduce memory. Optional `energy_method='frangi'` or `'sato'` leverages scikit-image's Frangi or Sato filters for faster vesselness. |
| `get_vertices_V200.m` | `slavv-streamlit/src/vectorization_core.py:extract_vertices` | Approximate | Uses voxel-spacing-aware ellipsoidal structuring element and radius-aware volume exclusion, returning radii in pixel and micron units.
| `get_edges_V300.m` | `slavv-streamlit/src/vectorization_core.py:extract_edges` | Approximate | Edge tracing uses radius-scaled steps with full terminal detection (near-vertex, energy rise, bounds via floor-based `_in_bounds`, max steps), deduplicates self/duplicate edges, and follows ridges via perpendicular-gradient steering with voxel-size-aware central differences and physical-distance vertex checks. Directions are seeded via local Hessian orientation, or set `direction_method='uniform'` to use evenly distributed directions. Supports optional voxel-snapped steps via `discrete_tracing` for integer-valued paths. Outputs standardized to NumPy arrays for traces and vertex connections; regression fixture locks synthetic-volume outputs for parity validation. |
| `get_edges_by_watershed.m` | `slavv-streamlit/src/vectorization_core.py:extract_edges_watershed` | Approximate | Watershed segmentation seeded at vertices grows regions and records boundary voxels where regions meet as edge traces; respects `energy_sign` and is validated with a regression-style test. |
| `get_network_V190.m` | `slavv-streamlit/src/vectorization_core.py:construct_network` | Approximate | Builds symmetric adjacency and strands, deduplicates edges with stable keying, removes short hairs, prunes cycles, tracks orphans, and retains dangling-edge traces. |

### Helpers and Subroutines

| MATLAB | Python | Parity | Notes |
|---|---|---|---|
| `energy_filter_V200.m`, `get_filter_kernel.m` | Integrated in `calculate_energy_field` | Approximate | Kernel construction simplified; PSF handling partially implemented and anisotropic smoothing applied. |
| `construct_structuring_element*.m` | Integrated in vertex extraction | Exact | Voxel-spacing-aware ellipsoidal structuring element matches MATLAB behavior.
| `get_vessel_directions_V2/V3/V5.m` | `slavv-streamlit/src/vectorization_core.py:_estimate_vessel_directions` | Approximate | Radius-aware local Hessian eigenvectors seed edges while respecting voxel spacing; falls back to uniform directions if ill-conditioned. |
| `get_chunking_lattice_V190.m` | `slavv-streamlit/src/vectorization_core.py:get_chunking_lattice` | Approximate | Generates overlapping z-axis chunks when volumes exceed `max_voxels_per_node_energy`.
| `pre_processing.m`, `fix_intensity_bands.m` | `slavv-streamlit/src/vectorization_core.py:preprocess_image` | Approximate | Intensity normalization with optional axial band correction via Gaussian smoothing.
| `vectorize_V200.m` parameter defaults | `slavv-streamlit/src/vectorization_core.py:validate_parameters` | Approximate | Applies MATLAB defaults and validates ranges with descriptive error messages.
| `combine_strands.m` | Integrated in `construct_network` | Approximate | Strand combining simplified. |
| `sort_network_V180.m`, `fix_strand_vertex_mismatch*.m` | Integrated in `construct_network` | Approximate | Strands sorted and mismatches flagged. |
| `clean_edges*.m` (hairs/orphans/cycles) | Integrated in `construct_network` | Approximate | Removes short hairs, identifies orphans, and prunes cycles. |
| Cropping: `crop_vertices_V200.m`, `crop_edges_V200.m`, `crop_vertices_by_mask.m` | `slavv-streamlit/src/vectorization_core.py:crop_vertices`, `crop_edges`, `crop_vertices_by_mask` | Approximate | Bounding-box and mask-based vertex/edge cropping helpers.

### Machine Learning Curation

| MATLAB | Python | Parity | Notes |
|---|---|---|---|
| `vertexCuratorNetwork_V*.m`, `edgeCuratorNetwork_V*.m` | `slavv-streamlit/src/ml_curator.py` | Approximate | Default logistic MLP classifiers mimic MATLAB curator networks; features and weights still differ.
| `choose_vertices_V200.m`, `choose_edges_V200.m` | `slavv-streamlit/src/ml_curator.py` | Approximate | Threshold-based selection via `choose_vertices` and `choose_edges` helpers.
| `vertex_curator.m`, `edge_curator.m` | `slavv-streamlit/src/ml_curator.py:AutomaticCurator` | Approximate | Heuristic vertex and edge curation without ML models.
| `vertex_feature_extractor.m`, `edge_info_extractor.m` | `slavv-streamlit/src/ml_curator.py` | Approximate | Feature sets expanded with radius/scale ratios, energy ratios, and endpoint radii statistics.
| `uncuratedInfoExtractor.m` | `slavv-streamlit/src/ml_curator.py:extract_uncurated_info` | Approximate | Extracts vertex and edge features for QA datasets without classification.
| `MLTraining.py`, `MLLibrary.py` | `slavv-streamlit/src/ml_curator.py`, `slavv-streamlit/app.py` | Approximate | CSV-driven training workflow for vertex and edge classifiers.

### Visualization

| MATLAB | Python | Parity | Notes |
|---|---|---|---|
| `visualize_vertices_V200.m`, `visualize_edges_V180.m`, `visualize_strands*.m` | `slavv-streamlit/src/visualization.py` | Approximate | 2D/3D Plotly visualizations implemented; styling/options differ.
| `visualize_depth_via_color_V200.m` | `slavv-streamlit/src/visualization.py:plot_2d_network`, `plot_3d_network` | Approximate | Edges colored by depth, energy, strand ID, radius, or length using Plotly colormaps with optional depth-based opacity, legends, and colorbars; 2D projections maintain equal axis scaling.
| `visualize_network_slice.m` | `slavv-streamlit/src/visualization.py:plot_network_slice` | Approximate | 2D slice along arbitrary axis with thickness filter and equal axis scaling.
| `animate_strands_3D.m` | `slavv-streamlit/src/visualization.py:animate_strands_3d` | Approximate | Animates strands sequentially with Plotly frames.
| `render_flow_field_V3/V4.m` | `slavv-streamlit/src/visualization.py:plot_flow_field` | Approximate | Renders edge orientations as 3D cones.

### Export and I/O

| MATLAB | Python | Parity | Notes |
|---|---|---|---|
| `vmv_mat2file.m`, `strand2vmv.m` | `slavv-streamlit/src/visualization.py:_export_vmv` | Approximate | Simplified VMV writer; not spec-complete.
| `strand2casx.m`, `casx_mat2file.m` | `slavv-streamlit/src/visualization.py:_export_casx` | Approximate | Minimal CASX XML writer.
| `casx_file2mat.m` | `slavv-streamlit/src/io_utils.py:load_network_from_casx` | Approximate | Basic CASX network import.
| `vmv_file2mat.m` | `slavv-streamlit/src/io_utils.py:load_network_from_vmv` | Approximate | Basic VMV network import.
| `casX2mat.m` | `slavv-streamlit/src/io_utils.py:load_network_from_mat` | Approximate | Basic MAT network import.
| *(no direct MATLAB equivalent)* | `slavv-streamlit/src/io_utils.py:load_network_from_csv`, `load_network_from_json`, `save_network_to_csv`, `save_network_to_json` | Approximate | Load or save network data using CSV or JSON files.
| MATLAB `save`/custom MAT writers | `slavv-streamlit/src/visualization.py:_export_mat` | Approximate | Export network data to MATLAB `.mat` files via `scipy.io.savemat`.
| `mat2tif.m`, `tif2mat.m`, `h52mat.m`, `mat2h5.m` | Python libs (`tifffile`, `h5py`) | Approximate | Standard Python I/O replaces MATLAB utilities; not 1:1.

### Statistics and Analysis

| MATLAB | Python | Parity | Notes |
|---|---|---|---|
| `calculate_network_statistics.m` | `slavv-streamlit/src/vectorization_core.py:calculate_network_statistics` | Approximate | Computes counts, strand and edge lengths, radii statistics, tortuosity, branching angles, edge-energy and edge-radius means/std, and volume/length/surface-area densities via `calculate_surface_area` and `calculate_vessel_volume`. |

### Notes

- Parity levels reflect current implementation state; see `docs/PORTING_SUMMARY.md` and `TODO.md` for planned improvements.
- File/function names reference their locations exactly: Python paths like `slavv-streamlit/src/vectorization_core.py` and MATLAB functions as in `Vectorization-Public/source/`.
