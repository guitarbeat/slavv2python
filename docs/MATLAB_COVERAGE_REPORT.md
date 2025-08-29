## MATLAB Coverage Report

This report inventories MATLAB `.m` files in `Vectorization-Public/` and checks for a Python counterpart based on `docs/MATLAB_TO_PYTHON_MAPPING.md` and the current Python modules under `slavv-streamlit/src/`.

Notes:
- "Mapped" means the MATLAB function/script is explicitly listed in the mapping doc or covered by an aggregated family entry (e.g., `construct_structuring_element*.m`).
- Many MATLAB files are example or study scripts (`vectorization_script_*`, `*_script*.m`, `noise_sensitivity_*`, `test_*`). These are intentionally not 1:1 ported; their functionality is subsumed by the app and tests.
- Several helpers are implemented indirectly inside Python modules (e.g., cleaning/cropping integrated in `vectorization_core.py`). Where not explicitly listed in the mapping doc, they appear as "Unmapped" below and are candidates to document or implement as needed.

### Summary

- Total MATLAB files scanned: 152
- Mapped by doc (explicit or family): 65
- Unmapped by doc: 87

See also: `docs/MATLAB_TO_PYTHON_MAPPING.md` and `docs/PORTING_SUMMARY.md`.

### Unmapped MATLAB Files by Category

The following `.m` files do not yet have an explicit mapping in `MATLAB_TO_PYTHON_MAPPING.md`. Some are out-of-scope scripts; some are helpers likely integrated or pending. This list is a checklist to either (a) document their coverage more explicitly, or (b) implement equivalents if required.

#### Scripts/Examples (27)
- animate_strands_3D_script.m
- for_Chakameh_vascular_vector_rendering_V600.m
- noise_sensitivity_study.m
- noise_sensitivity_study_V2.m
- register_strands_script.m
- registration_script_1D_example.m
- registration_script_test.m
- test_random_anatomy_generation.m
- test_strand_casX_conversion.m
- vectorization_script_2017MMDD_TxRed_chronic.m
- vectorization_script_Alankrit.m
- vectorization_script_Anna.m
- vectorization_script_Annie.m
- vectorization_script_Annie_2.m
- vectorization_script_Annie_3.m
- vectorization_script_Annie_4.m
- vectorization_script_Annie_5.m
- vectorization_script_Blinder.m
- vectorization_script_Chakameh_DVD.m
- vectorization_script_Chakameh_OCT.m
- vectorization_script_Dafna.m
- vectorization_script_Linninger.m
- vectorization_script_MGB_Broderick.m
- vectorization_script_MGB_David.m
- vectorization_script_MGB_David_DVD.m
- vectorization_script_Shaun.m
- vectorization_script_michael.m

#### Visualization/Plotting (6)
- area_histogram_plotter.m
- length_histogram_plotter.m
- network_histogram_plotter.m
- paint_vertex_image.m
- save_figures.m
- visualize_edges_annuli.m

#### I/O & Formats (8)
- casx2strand.m
- casx2vmv.m
- dicom2tif.m
- import_from_LPPD.m
- input_from_LPPD.m
- output_to_LPPD.m
- partition_casx_by_xy_bins.m
- registration_txt2mat.m

#### ML/Curation (3)
- edge_curator_Drews.m
- getTrainingArray.m
- simpleFeatureArray.m

#### Core/Helpers (43)
- add_vertices_to_edges.m
- calculate_center_of_area.m
- calculate_depth_statistics.m
- calculate_image_statistics_from_binary.m
- calculate_image_stats.m
- calculate_linear_strel.m
- calculate_linear_strel_range.m
- clean_edge_pairs.m
- clean_edges_cycles.m
- clean_edges_hairs.m
- clean_edges_orphans.m
- clean_edges_vertex_degree_excess.m
- evaluate_registration.m
- export_strand_data.m
- find_number_after_literal.m
- fix_strand_vertex_mismatch_again.m
- flow_field_subroutine.m
- fourier_transform_V2.m
- gaussian_blur.m
- gaussian_blur_in_chunks.m
- generate_reference_image.m
- getVertexDerivatives.m
- get_edge_metric.m
- get_edge_vectors.m
- get_edge_vectors_V300.m
- get_edges_by_watershed_method_one.m
- get_edges_for_vertex.m
- get_starts_and_counts_V200.m
- get_strand_objects.m
- index2position.m
- kstest_wrapper.m
- make_mask_from_registration.m
- randomize_anatomy.m
- register_strands.m
- register_vector_sets.m
- resample_vectors.m
- smooth_edges.m
- smooth_edges_V2.m
- smooth_hist.m
- sort_edges.m
- subsample_vectors.m
- transform_vector_set.m
- vertex_info_extractor.m

### Recommendations

- Document integrated helpers explicitly: add lines to the mapping doc for items already covered indirectly (e.g., cleaning, sorting, structuring-element variants, direction variants). This will reduce the "Unmapped" count without code changes.
- Clarify scope: mark `vectorization_script_*`, study/test scripts, and per-experiment driver scripts as intentionally not ported.
- Prioritize remaining helpers by impact: e.g., `get_edge_vectors*`, `*_registration*`, `*_hist*` could be (a) added as utilities, or (b) kept out-of-scope if the app covers the workflows.

