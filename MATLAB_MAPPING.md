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
| `energy_filter_V200.m` | `core/energy.py` | ✅ | Multi-scale Hessian energy |
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
| `choose_vertices_V200.m` | `core/energy.py` (vertex extraction) | 🔀 | |
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
| `choose_edges_V200.m` | `core/tracing.py` | ✅ | |
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
| `area_histogram_plotter.m` | — | ⬜ | Plotting utility |
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
| ✅ Ported | ~30 |
| 🔀 Merged | ~40 |
| ⬜ Not yet ported | ~10 |
| 🚫 Skipped | ~20+ scripts |

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
