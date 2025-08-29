## MATLAB → Python Mapping (Parity Levels)

This document maps key MATLAB SLAVV functions/scripts to their Python counterparts in this repository, and indicates the current parity level:
- Exact: Equivalent behavior intended and validated
- Approximate: Implemented and functional but not numerically identical or missing some heuristics/options
- Omitted: Not implemented yet (planned)

This document also summarizes coverage and known parity deviations to keep docs minimal.

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
| `calculate_linear_strel.m`, `calculate_linear_strel_range.m` | Integrated in vertex extraction | Approximate | Linear/spherical structuring variants handled via spacing-aware footprints. |
| `get_vessel_directions_V2/V3/V5.m` | `slavv-streamlit/src/vectorization_core.py:_estimate_vessel_directions` | Approximate | Radius-aware local Hessian eigenvectors seed edges while respecting voxel spacing; falls back to uniform directions if ill-conditioned. |
| `get_edge_vectors.m`, `get_edge_vectors_V300.m` | `slavv-streamlit/src/vectorization_core.py:_generate_edge_directions`, `_estimate_vessel_directions` | Approximate | Uniform spherical directions (Fibonacci) and Hessian-based directions cover seeding strategies. |
| `get_chunking_lattice_V190.m` | `slavv-streamlit/src/vectorization_core.py:get_chunking_lattice` | Approximate | Generates overlapping z-axis chunks when volumes exceed `max_voxels_per_node_energy`.
| `pre_processing.m`, `fix_intensity_bands.m` | `slavv-streamlit/src/vectorization_core.py:preprocess_image` | Approximate | Intensity normalization with optional axial band correction via Gaussian smoothing.
| `vectorize_V200.m` parameter defaults | `slavv-streamlit/src/vectorization_core.py:validate_parameters` | Approximate | Applies MATLAB defaults and validates ranges with descriptive error messages.
| `combine_strands.m` | Integrated in `construct_network` | Approximate | Strand combining simplified. |
| `sort_network_V180.m`, `fix_strand_vertex_mismatch*.m` | Integrated in `construct_network` | Approximate | Strands sorted and mismatches flagged. |
| `clean_edges*.m` (hairs/orphans/cycles/vertex_degree_excess), `clean_edge_pairs.m` | Integrated in `construct_network` | Approximate | Removes short hairs, identifies orphans, prunes cycles, and resolves small edge inconsistencies. |
| `sort_edges.m` | Integrated in `construct_network` | Approximate | Edge reordering/deduplication handled via stable keying and adjacency checks. |
| Cropping: `crop_vertices_V200.m`, `crop_edges_V200.m`, `crop_vertices_by_mask.m` | `slavv-streamlit/src/vectorization_core.py:crop_vertices`, `crop_edges`, `crop_vertices_by_mask` | Approximate | Bounding-box and mask-based vertex/edge cropping helpers. |
| `weighted_KStest2.m` | `slavv-streamlit/src/utils.py:weighted_ks_test` | Exact | Weighted two-sample Kolmogorov–Smirnov statistic. |
| `gaussian_blur.m`, `gaussian_blur_in_chunks.m` | `slavv-streamlit/src/vectorization_core.py:preprocess_image` | Approximate | Uses `scipy.ndimage.gaussian_filter` with optional chunking via the energy lattice when large volumes are present. |
| `flow_field_subroutine.m` | `slavv-streamlit/src/visualization.py:plot_flow_field` | Approximate | Renders edge orientations as 3D cones centered on midpoints of traces. |
| `get_edges_for_vertex.m` | `slavv-streamlit/src/vectorization_core.py:get_edges_for_vertex` | Exact | Returns indices of incident edges given an adjacency list of connections. |
| `get_edge_metric.m` | `slavv-streamlit/src/vectorization_core.py:get_edge_metric` | Approximate | Supports length and energy-based aggregates (mean/min/max/median). |
| `resample_vectors.m` | `slavv-streamlit/src/vectorization_core.py:resample_vectors` | Approximate | Resamples a polyline to near-uniform spacing by arc length. |
| `smooth_edges.m`, `smooth_edges_V2.m` | `slavv-streamlit/src/vectorization_core.py:smooth_edge_traces` | Approximate | 1D Gaussian smoothing along each coordinate sequence. |
| `transform_vector_set.m` | `slavv-streamlit/src/vectorization_core.py:transform_vector_set` | Approximate | Applies 4x4 homogeneous or scale/rotate/translate transforms to positions. |
| `subsample_vectors.m` | `slavv-streamlit/src/vectorization_core.py:subsample_vectors` | Exact | Keeps every Nth point, preserving endpoints. |
| `register_vector_sets.m` | `slavv-streamlit/src/vectorization_core.py:register_vector_sets` | Approximate | Rigid (Kabsch, optional scale) and affine least‑squares 3D registration; returns 4x4 transform and optional RMS error. |
| `register_strands.m` | `slavv-streamlit/src/vectorization_core.py:register_strands` | Approximate | Rigid (ICP) or affine alignment of two networks with vertex merge by distance; returns merged network and transform. |

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
| `visualize_strands_via_color_3D_V2/V3.m` | `slavv-streamlit/src/visualization.py:plot_3d_network` | Approximate | Strand coloring supported in 3D using Plotly `Set3` palette; colors differ from MATLAB.
| *(no direct MATLAB equivalent)* | `slavv-streamlit/src/visualization.py:plot_network_slice` | Approximate | 2D slice along arbitrary axis with thickness filter and equal axis scaling.
| `animate_strands_3D.m` | `slavv-streamlit/src/visualization.py:animate_strands_3d` | Approximate | Animates strands sequentially with Plotly frames.
| `render_flow_field_V3/V4.m` | `slavv-streamlit/src/visualization.py:plot_flow_field` | Approximate | Renders edge orientations as 3D cones.

### Export and I/O

| MATLAB | Python | Parity | Notes |
|---|---|---|---|
| `vmv_mat2file.m`, `strand2vmv.m` | `slavv-streamlit/src/visualization.py:_export_vmv` | Approximate | Simplified VMV writer; not spec-complete.
| `strand2casx.m`, `casx_mat2file.m` | `slavv-streamlit/src/visualization.py:_export_casx` | Approximate | Minimal CASX XML writer.
| `casx_file2mat.m` | `slavv-streamlit/src/io_utils.py:load_network_from_casx` | Approximate | Basic CASX network import.
| *(no direct MATLAB equivalent)* | `slavv-streamlit/src/io_utils.py:load_network_from_vmv` | Approximate | Basic VMV network import.
| `casX2mat.m` | `slavv-streamlit/src/io_utils.py:load_network_from_mat` | Approximate | Basic MAT network import.
| `dicom2tif.m` | `slavv-streamlit/src/io_utils.py:dicom_to_tiff` | Approximate | Reads single/multi-frame DICOM or series, optional TIFF export with sorting and rescale.
| *(no direct MATLAB equivalent)* | `slavv-streamlit/src/io_utils.py:load_network_from_csv`, `load_network_from_json`, `save_network_to_csv`, `save_network_to_json` | Approximate | Load or save network data using CSV or JSON files.
| MATLAB `save`/custom MAT writers | `slavv-streamlit/src/visualization.py:_export_mat` | Approximate | Export network data to MATLAB `.mat` files via `scipy.io.savemat`.
| `mat2tif.m`, `tif2mat.m`, `h52mat.m`, `mat2h5.m` | Python libs (`tifffile`, `h5py`) | Approximate | Standard Python I/O replaces MATLAB utilities; not 1:1.

#### Conversion Workflows (Compositions)

Replicate MATLAB conversion scripts by composing loaders and exporters:
- CASX → VMV: `io_utils.load_network_from_casx` → `visualization._export_vmv`
- CASX → MAT: `io_utils.load_network_from_casx` → `visualization._export_mat`
- VMV → CASX: `io_utils.load_network_from_vmv` → `visualization._export_casx`
- VMV → MAT: `io_utils.load_network_from_vmv` → `visualization._export_mat`
- MAT → CASX/VMV: `io_utils.load_network_from_mat` → `visualization._export_casx` / `visualization._export_vmv`
- Any → CSV/JSON: `io_utils.save_network_to_csv` / `io_utils.save_network_to_json`

### Statistics and Analysis

| MATLAB | Python | Parity | Notes |
|---|---|---|---|
| `calculate_network_statistics.m` | `slavv-streamlit/src/vectorization_core.py:calculate_network_statistics` | Approximate | Computes counts, strand and edge lengths, radii statistics, tortuosity, branching angles, edge-energy and edge-radius means/std, volume/length/surface-area/vertex/edge densities via `calculate_surface_area` and `calculate_vessel_volume`, vertex-degree statistics, and graph connectivity metrics (components, endpoints, average path length, clustering coefficient, diameter, betweenness centrality, closeness centrality, eigenvector centrality, graph density). |

### Coverage Summary

- Total MATLAB `.m` files scanned: 152
- Mapped by this document (explicit or family): 109
- Unmapped (by doc): 43 — largely example/study scripts and peripheral helpers

Unmapped categories (representative examples):
- Scripts/Examples: `vectorization_script_*`, `*_script*.m`, `noise_sensitivity_*`, `test_*` (intentionally not ported).
- Visualization/Plotting: `*_histogram_plotter.m`, `visualize_edges_annuli.m`, `paint_vertex_image.m`.
- I/O & Formats: `dicom2tif.m`, `casx2*.m`, `partition_casx_by_xy_bins.m`, `registration_txt2mat.m`.
- ML/Curation: `edge_curator_Drews.m`, `getTrainingArray.m`, `simpleFeatureArray.m`.
- Core/Helpers: `get_edges_for_vertex.m`, `get_edge_metric.m`, `resample_vectors.m`, `smooth_edges*.m`, `transform_vector_set.m`, `register_strands.m`.

Notes:
- Many helpers are integrated into Python modules (e.g., cleaning/sorting/cropping/structuring). When in doubt, search this document for the family name.

### Parity Deviations (Rationale)

- Energy Field:
  - PSF weighting uses a simplified Gaussian approximation; MATLAB kernels are more detailed.
  - Filter ratios (Gaussian/annular) approximate defaults rather than exact numeric parity.
- Vertex and Edge Extraction:
  - Tracing uses floating-point updates by default; enable `discrete_tracing=True` for voxel-snapped steps.
  - Direction estimation falls back to uniform orientations when eigenanalysis is unstable.
- Visualization:
  - Plotly replaces MATLAB graphics; colormaps and camera controls differ.
  - Strand coloring uses Plotly palettes instead of MATLAB’s hardcoded tables.

### Notes

- Parity levels reflect current implementation state; for priorities see the repository issue tracker.

### FAQ / Glossary

- Axis order: Arrays use `(y, x, z)` indexing throughout. When converting to physical units, multiply by `microns_per_voxel = [µm_y, µm_x, µm_z]` element‑wise.
- Radii units: `radii_pixels` are scale-derived pixel radii; `radii_microns` are physical radii. Public APIs return both where relevant.
- Energy sign: Bright vessels default to negative energy (`energy_sign = -1.0`), so lower values are more vessel‑like. Flip the sign for dark vessels.
- Discrete tracing: Set `discrete_tracing=True` to snap edge steps to voxel centers (closer to MATLAB integer stepping). Default uses floating‑point steps for smoother paths.
- Direction seeding: `direction_method='hessian'` uses local Hessian eigenvectors; `'uniform'` uses evenly distributed unit vectors.
- Voxel anisotropy: All neighborhood and gradient ops account for anisotropic voxels via `microns_per_voxel`.
- File/function names reference their locations exactly: Python paths like `slavv-streamlit/src/vectorization_core.py` and MATLAB functions as in `Vectorization-Public/source/`.
