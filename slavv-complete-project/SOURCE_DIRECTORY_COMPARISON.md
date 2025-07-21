# Source Directory Comparison: Original MATLAB vs. Python/Streamlit SLAVV

This document provides a detailed comparison between the original `source` directory from the MATLAB-based SLAVV repository (`Vectorization-Public/source/`) and the `src` directory of the new Python/Streamlit implementation (`slavv-streamlit/src/`).

## Overview

The original MATLAB repository contains a large number of `.m` files, many of which are utility functions, specific experimental scripts, or older versions of core algorithms. The Python/Streamlit implementation consolidates the core functionality into a more modular and streamlined structure, focusing on the essential components required for the SLAVV algorithm and its user interface.

## Original MATLAB `source/` Directory Contents

```
MLDeployment.py
MLLibrary.py
MLTraining.py
README.md
add_vertices_to_edges.m
animate_strands_3D.m
animate_strands_3D_script.m
area_histogram_plotter.m
calculate_center_of_area.m
calculate_depth_statistics.m
calculate_image_statistics_from_binary.m
calculate_image_stats.m
calculate_linear_strel.m
calculate_linear_strel_range.m
calculate_network_statistics.m
calculate_surface_area.m
casX2mat.m
casx2strand.m
casx2vmv.m
casx_file2mat.m
casx_mat2file.m
choose_edges_V200.m
choose_vertices_V200.m
clean_edge_pairs.m
clean_edges.m
clean_edges_cycles.m
clean_edges_hairs.m
clean_edges_orphans.m
clean_edges_vertex_degree_excess.m
combine_strands.m
construct_structuring_element.m
construct_structuring_element_V190.m
construct_structuring_elements.m
crop_edges_V200.m
crop_vertices_V200.m
crop_vertices_by_mask.m
dicom2tif.m
edgeCuratorNetwork_V1.m
edgeCuratorNetwork_V2.m
edgeCuratorNetwork_V4_20.m
edge_curator.m
edge_curator_Drews.m
edge_info_extractor.m
energy_filter_V200.m
energy_filter_V200_backup_191202.m
evaluate_registration.m
export_strand_data.m
find_number_after_literal.m
fix_intensity_bands.m
fix_strand_vertex_mismatch.m
fix_strand_vertex_mismatch_again.m
flow_field_subroutine.m
for_Chakameh_vascular_vector_rendering_V600.m
fourier_transform_V2.m
gaussian_blur.m
gaussian_blur_in_chunks.m
generate_reference_image.m
getTrainingArray.m
getVertexDerivatives.m
get_chunking_lattice_V190.m
get_edge_metric.m
get_edge_vectors.m
get_edge_vectors_V300.m
get_edges_V203.m
get_edges_V204.m
get_edges_V300.m
get_edges_by_watershed.m
get_edges_by_watershed_method_one.m
get_edges_for_vertex.m
get_energy_V202.m
get_filter_kernel.m
get_network_V190.m
get_starts_and_counts_V200.m
get_strand_objects.m
get_vertices_V200.m
get_vessel_directions_V2.m
get_vessel_directions_V3.m
get_vessel_directions_V5.m
h52mat.m
import_from_LPPD.m
index2position.m
input_from_LPPD.m
kstest_wrapper.m
length_histogram_plotter.m
make_mask_from_registration.m
mat2h5.m
mat2tif.m
network_histogram_plotter.m
noise_sensitivity_study.m
noise_sensitivity_study_V2.m
output_to_LPPD.m
paint_vertex_image.m
partition_casx_by_xy_bins.m
pre_processing.m
pythonSetup.py
randomize_anatomy.m
register_strands.m
register_strands_script.m
register_vector_sets.m
registration_script_1D_example.m
registration_script_test.m
registration_txt2mat.m
render_flow_field_V3.m
render_flow_field_V4.m
resample_vectors.m
save_figures.m
simpleFeatureArray.m
smooth_edges.m
smooth_edges_V2.m
smooth_hist.m
sort_edges.m
sort_network_V180.m
strand2casx.m
strand2vmv.m
subsample_vectors.m
test_random_anatomy_generation.m
test_strand_casX_conversion.m
tif2mat.m
transform_vector_set.m
uncuratedInfoExtractor.m
vectorization_script_2017MMDD_TxRed_chronic.m
vectorization_script_Alankrit.m
vectorization_script_Anna.m
vectorization_script_Annie.m
vectorization_script_Annie_2.m
vectorization_script_Annie_3.m
vectorization_script_Annie_4.m
vectorization_script_Annie_5.m
vectorization_script_Blinder.m
vectorization_script_Chakameh_DVD.m
vectorization_script_Chakameh_OCT.m
vectorization_script_Dafna.m
vectorization_script_Linninger.m
vectorization_script_MGB_Broderick.m
vectorization_script_MGB_David.m
vectorization_script_MGB_David_DVD.m
vectorization_script_Shaun.m
vectorization_script_michael.m
vectorize_V190_20170315_MouseT326_session2_fused.m
vectorize_V200.m
vertexCuratorNetwork_V1.m
vertexCuratorNetwork_V2.m
vertexCuratorNetwork_V3.m
vertex_curator.m
vertex_feature_extractor.m
vertex_info_extractor.m
visualize_depth_via_color_V200.m
visualize_edges_V180.m
visualize_edges_annuli.m
visualize_strands.m
visualize_strands_via_color_3D_V2.m
visualize_strands_via_color_3D_V3.m
visualize_strands_via_color_V2.m
visualize_strands_via_color_V200.m
visualize_vertices_V200.m
vmv_mat2file.m
weighted_KStest2.m
```

## Python/Streamlit `slavv-streamlit/src/` Directory Contents

```
__pycache__/
ml_curator.py
vectorization_core.py
visualization.py
```

## Mapping of Functionality

The Python/Streamlit implementation consolidates the core logic from numerous MATLAB files into three main modules, plus the main `app.py` for the Streamlit interface.

### `vectorization_core.py`
This module is the direct Python equivalent of the core SLAVV algorithm, primarily porting the functionality found in `vectorize_V200.m` and its direct dependencies. It encompasses the four main steps of the vectorization process:

*   **Energy Image Formation**: Functionality from `energy_filter_V200.m`, `get_filter_kernel.m`, and related PSF approximation (`approximating_PSF` logic in `vectorize_V200.m`) is integrated here.
*   **Vertex Extraction**: Logic from `get_vertices_V200.m` and related local minima detection and volume exclusion is implemented.
*   **Edge Extraction**: Functionality from `get_edges_V203.m`, `get_edges_V204.m`, `get_edges_V300.m`, and the edge tracing logic is included.
*   **Network Construction**: The assembly of strands and network components, drawing from `get_network_V190.m` and `combine_strands.m`, is handled here.
*   **Parameter Handling**: The detailed parameter definitions and validation, originally found in the comments and logic of `vectorize_V200.m`, are now part of this module's parameter validation functions.

### `ml_curator.py`
This module is responsible for the machine learning-assisted curation of vertices and edges. It directly ports and enhances the functionality found in the original Python files within the MATLAB repository:

*   `MLDeployment.py`: The core logic for applying trained ML models for curation.
*   `MLLibrary.py`: Contains the feature extraction methods and classification algorithms used for both vertex and edge curation.
*   `MLTraining.py`: While not directly part of the runtime `src` directory, the concepts for training data generation and model persistence from `MLTraining.py` are considered in the design of the `MLCurator` for future training capabilities.
*   Related MATLAB curation scripts like `choose_edges_V200.m`, `choose_vertices_V200.m`, `edgeCuratorNetwork_V*.m`, `vertexCuratorNetwork_V*.m`, `edge_curator.m`, `vertex_curator.m`, `vertex_feature_extractor.m`, `edge_info_extractor.m`, and `uncuratedInfoExtractor.m` have their core logic and concepts integrated into `ml_curator.py` to provide a unified ML curation interface.

### `visualization.py`
This module handles all aspects of visualizing the vectorized network. It consolidates the functionality from various MATLAB visualization scripts:

*   `visualize_depth_via_color_V200.m`, `visualize_edges_V180.m`, `visualize_strands.m`, `visualize_strands_via_color_3D_V2.m`, `visualize_strands_via_color_3D_V3.m`, `visualize_strands_via_color_V2.m`, `visualize_strands_via_color_V200.m`, `visualize_vertices_V200.m`: The core plotting and rendering logic from these files is translated into Plotly-based visualizations.
*   `animate_strands_3D.m`, `animate_strands_3D_script.m`: Concepts for 3D rendering and interactive camera controls are incorporated into the Plotly visualizations.
*   `area_histogram_plotter.m`, `length_histogram_plotter.m`, `network_histogram_plotter.m`: The statistical plotting capabilities are integrated into the `visualization.py` and `app.py` for the Analysis page.
*   `render_flow_field_V3.m`, `render_flow_field_V4.m`: While not fully implemented in the current version, the framework for rendering flow fields is considered for future expansion.

## Files Not Directly Ported (and why)

Many files in the original MATLAB `source/` directory were not directly translated one-to-one into Python modules. This is primarily due to:

*   **Redundancy/Older Versions**: Many files (e.g., `vectorize_V190_*.m`, `energy_filter_V200_backup_*.m`, multiple versions of `get_edges_V*.m`, `vertexCuratorNetwork_V*.m`) represent older iterations or backups of the core algorithm. The Python implementation focuses on the latest and most robust version of the algorithm.
*   **Specific Experimental Scripts**: Files like `vectorization_script_*.m` are likely tailored for specific experimental setups or datasets. The Python/Streamlit app provides a generalized interface for processing, making these specific scripts unnecessary.
*   **MATLAB-Specific Utilities**: Files related to MATLAB's internal data handling (`h52mat.m`, `mat2h5.m`, `mat2tif.m`, `tif2mat.m`, `vmv_mat2file.m`, `casX2mat.m`, `casx_file2mat.m`, `casx_mat2file.m`) are replaced by Python's native file I/O and libraries like `tifffile`, `h5py`, and custom parsers for VMV/CASX.
*   **Minor Helper Functions**: Many small MATLAB functions (e.g., `find_number_after_literal.m`, `index2position.m`, `smooth_hist.m`) are either absorbed directly into the main Python functions, replaced by standard Python library calls, or deemed unnecessary in the new architecture.
*   **Unimplemented Features (from MATLAB README)**: Some features mentioned in the MATLAB `vectorize_V200.m` README (e.g., `NetworkPath` parameter for training networks) were marked as 

