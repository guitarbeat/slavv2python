# MATLAB → Python Mapping

> Cross-reference between files in `external/Vectorization-Public/source/` (MATLAB)
> and `source/slavv/` (Python).  Updated after the refactoring on 2026-03-04.

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Ported — Python equivalent exists |
| 🔀 | Merged — absorbed into a larger Python module |
| ⬜ | Not yet ported |
| 🚫 | Intentionally skipped (script/demo/obsolete) |

---

## 1. Pipeline Orchestration

| MATLAB File | Python File | Status | Notes |
|---|---|---|---|
| `vectorize_V200.m` | `core/pipeline.py` (`SLAVVProcessor`) | ✅ | Main orchestrator |

---

## 2. Energy / Enhancement

| MATLAB File | Python File | Status | Notes |
|---|---|---|---|
| `energy_filter_V200.m` | `core/energy.py` | 🔀 | Core port exists, but real-volume parity is still blocked by octave/downsampling and projected scale-index differences. |
| `energy_filter_V200_backup_191202.m` | — | 🚫 | Old backup |
| `construct_structuring_element.m` | `core/energy.py` (`construct_structuring_element`) | 🔀 | Merged into energy |
| `construct_structuring_element_V190.m` | — | 🚫 | Superseded by above. V200 improved grid spacing logic with generalized ellipsoid generation. |
| `construct_structuring_elements.m` | `core/energy.py` | 🔀 | |
| `calculate_linear_strel.m` | `core/energy.py` | 🔀 | |
| `calculate_linear_strel_range.m` | `core/energy.py` | 🔀 | |
| `get_energy_V200.m` | `core/energy.py` | 🔀 | |

---

## 3. Vertex Detection

| MATLAB File | Python File | Status | Notes |
|---|---|---|---|
| `choose_vertices_V200.m` | `core/tracing.py` (vertex extraction) | 🔀 | MATLAB-style crop/paint selection is ported, but exact parity still depends on upstream energy parity. |
| `vertex_curator.m` | `visualization/interactive_curator.py`, `analysis/ml_curator.py` | ✅ | GCI + ML curator |
| `vertex_info_extractor.m` | `io/matlab_parser.py` (`extract_vertices`) | ✅ | |
| `vertex_feature_extractor.m` | `analysis/ml_curator.py` | 🔀 | |
| `vertexCuratorNetwork_V1.m` | `visualization/interactive_curator.py` | 🔀 | Superseded by V3. V1 lacked multi-scale support and refined labeling controls. |
| `vertexCuratorNetwork_V2.m` | `visualization/interactive_curator.py` | 🔀 | Superseded by V3. V2 improved UI, but V3 finalized Graph extraction and advanced representations. |
| `vertexCuratorNetwork_V3.m` | `visualization/interactive_curator.py` | ✅ | Latest GCI version |
| `uncuratedInfoExtractor.m` | `analysis/ml_curator.py` | ✅ | Ported `extract_uncurated_info` method |

---

## 4. Edge Extraction & Tracing

| MATLAB File | Python File | Status | Notes |
|---|---|---|---|
| `choose_edges_V200.m` | `core/tracing.py` | 🔀 | Candidate cleanup exists, but final chosen-edge parity is not yet exact on the real comparison volume. |
| `get_edges_V200.m` | `core/tracing.py` | 🔀 | |
| `add_vertices_to_edges.m` | `core/tracing.py` | 🔀 | |
| `clean_edges.m` | `core/tracing.py` | 🔀 | |
| `clean_edges_cycles.m` | `core/tracing.py` | 🔀 | |
| `clean_edges_hairs.m` | `core/tracing.py` | 🔀 | |
| `clean_edges_orphans.m` | `core/tracing.py` | 🔀 | |
| `clean_edges_vertex_degree_excess.m` | `core/tracing.py` | 🔀 | |
| `clean_edge_pairs.m` | `core/tracing.py` | 🔀 | |
| `edge_curator.m` | `visualization/interactive_curator.py` | ✅ | |
| `edge_curator_Drews.m` | — | 🚫 | User-specific script |
| `edge_info_extractor.m` | `io/matlab_parser.py` (`extract_edges`) | ✅ | |
| `edgeCuratorNetwork_V1.m` | `visualization/interactive_curator.py` | 🔀 | Superseded by V4_20. V1 lacked advanced trace visualization. |
| `edgeCuratorNetwork_V2.m` | `visualization/interactive_curator.py` | 🔀 | Superseded by V4_20. V2 added basic features, but V4_20 integrated ML curation inputs and edge selection refinements. |
| `edgeCuratorNetwork_V4_20.m` | `visualization/interactive_curator.py` | ✅ | Latest |

---

## 5. Network / Graph Construction

| MATLAB File | Python File | Status | Notes |
|---|---|---|---|
| `get_network_V200.m` | `core/graph.py` | ✅ | |
| `combine_strands.m` | `core/graph.py` | 🔀 | |
| `fix_strand_vertex_mismatch.m` | `core/graph.py` | 🔀 | |
| `fix_strand_vertex_mismatch_again.m` | — | 🚫 | Hotfix, merged |
| `export_strand_data.m` | `io/exporters.py` | ✅ | |
| `calculate_network_statistics.m` | `analysis/geometry.py` | ✅ | |

---

## 6. Cropping / Region Tools

| MATLAB File | Python File | Status | Notes |
|---|---|---|---|
| `crop_vertices_V200.m` | `analysis/geometry.py` (`crop_vertices`) | ✅ | |
| `crop_edges_V200.m` | `analysis/geometry.py` (`crop_edges`) | ✅ | |
| `crop_vertices_by_mask.m` | `analysis/geometry.py` (`crop_vertices_by_mask`) | ✅ | |

---

## 7. Visualization

| MATLAB File | Python File | Status | Notes |
|---|---|---|---|
| `visualize_vertices_V200.m` | `visualization/network_plots.py` | ✅ | |
| `visualize_edges_V180.m` | `visualization/network_plots.py` | 🔀 | |
| `visualize_edges_annuli.m` | `visualization/network_plots.py` | 🔀 | |
| `visualize_strands.m` | `visualization/network_plots.py` | 🔀 | |
| `visualize_strands_via_color_V200.m` | `visualization/network_plots.py` | 🔀 | |
| `visualize_strands_via_color_V2.m` | `visualization/network_plots.py` | 🔀 | |
| `visualize_strands_via_color_3D_V2.m` | `visualization/network_plots.py` | 🔀 | |
| `visualize_strands_via_color_3D_V3.m` | `visualization/network_plots.py` | 🔀 | |
| `visualize_depth_via_color_V200.m` | `visualization/network_plots.py` | 🔀 | |
| `animate_strands_3D.m` | `visualization/network_plots.py` | 🔀 | |
| `animate_strands_3D_script.m` | — | 🚫 | Script, not function |
| `flow_field_subroutine.m` | `visualization/volume_rasterization.py` | ✅ | |

---

## 8. File I/O & Format Conversion

| MATLAB File | Python File | Status | Notes |
|---|---|---|---|
| `dicom2tif.m` | `io/tiff.py` (`dicom_to_tiff`) | ✅ | |
| `tif2mat.m` | `io/tiff.py` (`load_tiff_volume`) | ✅ | |
| `casX2mat.m` | `io/network_io.py` (`load_network_from_casx`) | ✅ | |
| `casx_file2mat.m` | `io/network_io.py` | 🔀 | |
| `casx_mat2file.m` | `io/network_io.py` | ✅ | Ported `save_network_to_casx` |
| `casx2strand.m` | `io/network_io.py` | 🔀 | |
| `casx2vmv.m` | `io/network_io.py` | ✅ | Ported `convert_casx_to_vmv` |
| `vmv_mat2file.m` | `io/network_io.py` | ✅ | Ported `save_network_to_vmv` |
| `registration_txt2mat.m` | `io/exporters.py` (`parse_registration_file`) | ✅ | |
| `partition_casx_by_xy_bins.m` | `io/exporters.py` (`partition_network`) | ✅ | |

---

## 9. Statistics & Analysis

| MATLAB File | Python File | Status | Notes |
|---|---|---|---|
| `calculate_surface_area.m` | `analysis/geometry.py` | ✅ | |
| `calculate_depth_statistics.m` | `analysis/geometry.py` | 🔀 | |
| `calculate_center_of_area.m` | `analysis/geometry.py` | 🔀 | |
| `calculate_image_statistics_from_binary.m` | `analysis/geometry.py` | 🔀 | Logic covered by `calculate_image_stats` |
| `calculate_image_stats.m` | `analysis/geometry.py` | ✅ | Ported `calculate_image_stats` |
| `area_histogram_plotter.m` | `visualization/network_plots.py` | ✅ | Ported `plot_length_weighted_histograms` |
| `weighted_KStest2.m` | `utils/math.py` (`weighted_ks_test`) | ✅ | |
| `fourier_transform_V2.m` | `utils/math.py` | ✅ | Ported `fourier_transform_even` |
| `fix_intensity_bands.m` | `utils/preprocessing.py` | 🔀 | |
| `evaluate_registration.m` | `analysis/geometry.py` | ✅ | Ported `evaluate_registration` |

---

## 10. Utility / Helper Functions

| MATLAB File | Python File | Status | Notes |
|---|---|---|---|
| `find_number_after_literal.m` | — | 🚫 | MATLAB string parsing |
| `for_Chakameh_vascular_vector_rendering_V600.m` | — | 🚫 | User-specific |
| `get_vessel_directions_V3.m` | `analysis/geometry.py` | 🔀 | |

---

## 11. User-specific Vectorization Scripts

All `vectorization_script_*.m` files are user-specific example scripts and are **intentionally not ported** (🚫).

---

## Summary

| Status | Count |
|--------|-------|
| ✅ Ported | 31 |
| 🔀 Merged | 38 |
| ⬜ Not yet ported | 0 |
| 🚫 Skipped | 106 (including example scripts and obsolete files) |

### Key Gaps (✅ Addressed)

| MATLAB File | Priority | Notes |
|---|---|---|
| `casx_mat2file.m` | Medium | ✅ Implemented CASX **writer** |
| `vmv_mat2file.m` | Low | ✅ Implemented VMV **writer** |
| `casx2vmv.m` | Low | ✅ Implemented cross-format converter |
| `calculate_image_stats.m` | Medium | ✅ Implemented image-level statistics |
| `fourier_transform_V2.m` | Low | ✅ Implemented spectral analysis |
| `evaluate_registration.m` | Medium | ✅ Implemented registration evaluation |
| `uncuratedInfoExtractor.m` | Medium | ✅ Implemented pre-curation info |


## 12. Unmapped / Obsolete Scripts

These files were present in the MATLAB source but never mapped. They are considered obsolete or user-specific scripts.

| MATLAB File | Python File | Status | Notes |
|---|---|---|---|
| `gaussian_blur.m` | — | 🚫 | Unmapped/Obsolete |
| `gaussian_blur_in_chunks.m` | — | 🚫 | Unmapped/Obsolete |
| `generate_reference_image.m` | — | 🚫 | Unmapped/Obsolete |
| `getTrainingArray.m` | — | 🚫 | Unmapped/Obsolete |
| `getVertexDerivatives.m` | — | 🚫 | Unmapped/Obsolete |
| `get_chunking_lattice_V190.m` | — | 🚫 | Unmapped/Obsolete |
| `get_edge_metric.m` | — | 🚫 | Unmapped/Obsolete |
| `get_edge_vectors.m` | — | 🚫 | Unmapped/Obsolete |
| `get_edge_vectors_V300.m` | — | 🚫 | Unmapped/Obsolete |
| `get_edges_V203.m` | — | 🚫 | Unmapped/Obsolete |
| `get_edges_V204.m` | — | 🚫 | Unmapped/Obsolete |
| `get_edges_V300.m` | — | 🚫 | Unmapped/Obsolete |
| `get_edges_by_watershed.m` | — | 🚫 | Unmapped/Obsolete |
| `get_edges_by_watershed_method_one.m` | — | 🚫 | Unmapped/Obsolete |
| `get_edges_for_vertex.m` | — | 🚫 | Unmapped/Obsolete |
| `get_energy_V202.m` | — | 🚫 | Unmapped/Obsolete |
| `get_filter_kernel.m` | — | 🚫 | Unmapped/Obsolete |
| `get_network_V190.m` | — | 🚫 | Unmapped/Obsolete |
| `get_starts_and_counts_V200.m` | — | 🚫 | Unmapped/Obsolete |
| `get_strand_objects.m` | — | 🚫 | Unmapped/Obsolete |
| `get_vertices_V200.m` | — | 🚫 | Unmapped/Obsolete |
| `get_vessel_directions_V2.m` | — | 🚫 | Unmapped/Obsolete |
| `get_vessel_directions_V5.m` | — | 🚫 | Unmapped/Obsolete |
| `h52mat.m` | — | 🚫 | Unmapped/Obsolete |
| `import_from_LPPD.m` | — | 🚫 | Unmapped/Obsolete |
| `index2position.m` | — | 🚫 | Unmapped/Obsolete |
| `input_from_LPPD.m` | — | 🚫 | Unmapped/Obsolete |
| `kstest_wrapper.m` | — | 🚫 | Unmapped/Obsolete |
| `length_histogram_plotter.m` | — | 🚫 | Unmapped/Obsolete |
| `make_mask_from_registration.m` | — | 🚫 | Unmapped/Obsolete |
| `mat2h5.m` | — | 🚫 | Unmapped/Obsolete |
| `mat2tif.m` | — | 🚫 | Unmapped/Obsolete |
| `network_histogram_plotter.m` | — | 🚫 | Unmapped/Obsolete |
| `noise_sensitivity_study.m` | — | 🚫 | Unmapped/Obsolete |
| `noise_sensitivity_study_V2.m` | — | 🚫 | Unmapped/Obsolete |
| `output_to_LPPD.m` | — | 🚫 | Unmapped/Obsolete |
| `paint_vertex_image.m` | — | 🚫 | Unmapped/Obsolete |
| `pre_processing.m` | — | 🚫 | Unmapped/Obsolete |
| `randomize_anatomy.m` | — | 🚫 | Unmapped/Obsolete |
| `register_strands.m` | — | 🚫 | Unmapped/Obsolete |
| `register_strands_script.m` | — | 🚫 | Unmapped/Obsolete |
| `register_vector_sets.m` | — | 🚫 | Unmapped/Obsolete |
| `registration_script_1D_example.m` | — | 🚫 | Unmapped/Obsolete |
| `registration_script_test.m` | — | 🚫 | Unmapped/Obsolete |
| `render_flow_field_V3.m` | — | 🚫 | Unmapped/Obsolete |
| `render_flow_field_V4.m` | — | 🚫 | Unmapped/Obsolete |
| `resample_vectors.m` | — | 🚫 | Unmapped/Obsolete |
| `save_figures.m` | — | 🚫 | Unmapped/Obsolete |
| `simpleFeatureArray.m` | — | 🚫 | Unmapped/Obsolete |
| `smooth_edges.m` | — | 🚫 | Unmapped/Obsolete |
| `smooth_edges_V2.m` | — | 🚫 | Unmapped/Obsolete |
| `smooth_hist.m` | — | 🚫 | Unmapped/Obsolete |
| `sort_edges.m` | — | 🚫 | Unmapped/Obsolete |
| `sort_network_V180.m` | — | 🚫 | Unmapped/Obsolete |
| `strand2casx.m` | — | 🚫 | Unmapped/Obsolete |
| `strand2vmv.m` | — | 🚫 | Unmapped/Obsolete |
| `subsample_vectors.m` | — | 🚫 | Unmapped/Obsolete |
| `test_random_anatomy_generation.m` | — | 🚫 | Unmapped/Obsolete |
| `test_strand_casX_conversion.m` | — | 🚫 | Unmapped/Obsolete |
| `transform_vector_set.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_2017MMDD_TxRed_chronic.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_Alankrit.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_Anna.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_Annie.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_Annie_2.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_Annie_3.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_Annie_4.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_Annie_5.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_Blinder.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_Chakameh_DVD.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_Chakameh_OCT.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_Dafna.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_Linninger.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_MGB_Broderick.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_MGB_David.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_MGB_David_DVD.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_Shaun.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorization_script_michael.m` | — | 🚫 | Unmapped/Obsolete |
| `vectorize_V190_20170315_MouseT326_session2_fused.m` | — | 🚫 | Unmapped/Obsolete |

---

## 13. MATLAB Scripts (Non-Functions) Summary

During the review of the original MATLAB codebase, 34 files were identified as **scripts** (files that do not define a `function` at the top level). These scripts were primarily used for batch processing, testing, and utilities, and they are generally **intentionally not ported** to the core Python library, though their logic may inform future Python CLI tools or Jupyter notebooks.

### 13.1 User-Specific Vectorization Scripts (18 files)
*   **Files:** `vectorization_script_*.m` (e.g., `_Anna.m`, `_Annie.m`, `_Blinder.m`, etc.)
*   **Purpose:** These are hardcoded execution scripts tailored to specific users, datasets, or experiments. They set up input paths, configure pipeline parameters (e.g., `microns_per_voxel`, thresholds), and sequentially call `vectorize_V200` to run the vectorization workflow. 
*   **Python Equivalent:** Replaced by the modular `SLAVVProcessor` and generic entry points like the CLI or Jupyter notebooks.

### 13.2 Experiments & Studies (2 files)
*   **Files:** `noise_sensitivity_study.m`, `noise_sensitivity_study_V2.m`
*   **Purpose:** Evaluation scripts that systematically add synthetic noise to ground-truth images and assess the vectorization algorithm's robustness. They sweep through parameter ranges and compute ROC curves to find the best classifiers.

### 13.3 Test Scripts (2 files)
*   **Files:** `test_random_anatomy_generation.m`, `test_strand_casX_conversion.m`
*   **Purpose:** Validation scripts. One generates simulated vascular anatomy with randomized geometries and verifies the extracted network statistics, while the other verifies the integrity of CASX format conversion.

### 13.4 Registration Demos & Utilities (3 files)
*   **Files:** `register_strands_script.m`, `registration_script_1D_example.m`, `registration_script_test.m`
*   **Purpose:** Demonstration and testing scripts for registering/aligning different sets of vascular vectors. They apply test transformations and evaluate the goodness of registration.

### 13.5 Data Processing & Correction Utilities (9 files)
These are ad-hoc utilities for one-off tasks, data conversion, or hotfixing issues:
*   **`dicom2tif.m`**: Converts DICOM image sequences into multi-page TIFF stacks.
*   **`pre_processing.m`**: Applies background subtraction, normalization, and contrast enhancement to TIFFs prior to vectorization.
*   **`export_strand_data.m`**: Loads a curated network, filters strands by z-direction and radius, and exports the geometry.
*   **`fix_strand_vertex_mismatch_again.m`**: A hotfix script to clean up problematic networks by identifying and deleting strands with anomalously long segment lengths.
*   **`animate_strands_3D_script.m`**: Top-level script to trigger the 3D strand animation logic.
*   **Others**: `import_from_LPPD.m`, `for_Chakameh_vascular_vector_rendering_V600.m`, `kstest_wrapper.m`, `vectorize_V190_20170315_MouseT326_session2_fused.m`. 

*Note: The script `run_matlab_vectorization.m` is technically wrapped as a function to accept CLI arguments, but it serves the role of a non-interactive batch script and equivalent functionality is provided by the Python SLAVV CLI.*
