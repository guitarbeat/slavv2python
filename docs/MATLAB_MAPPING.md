# MATLAB to Python Mapping

Cross-reference between the upstream MATLAB sources in
`external/Vectorization-Public/source/` and the maintained Python modules in
`source/slavv/`. This document is intentionally focused on maintained surfaces
and parity-sensitive behavior rather than exhaustively listing every archival
MATLAB script.

## Status Labels

| Status | Meaning |
| --- | --- |
| `Ported` | A Python equivalent exists as a dedicated function or module |
| `Merged` | MATLAB behavior is absorbed into a broader Python module |
| `Skipped` | Script, demo, backup, or user-specific helper that is not part of the maintained Python package |

## Pipeline Orchestration

| MATLAB File | Python Location | Status | Notes |
| --- | --- | --- | --- |
| `vectorize_V200.m` | `source/slavv/core/pipeline.py` (`SLAVVProcessor`) | `Ported` | Main pipeline orchestration entry point |
| `run_matlab_vectorization.m` | `workspace/scripts/cli/compare_matlab_python.py` and package CLI | `Merged` | Batch-style orchestration now lives in Python wrappers and CLI workflows |

## Energy And Enhancement

| MATLAB File | Python Location | Status | Notes |
| --- | --- | --- | --- |
| `energy_filter_V200.m` | `source/slavv/core/energy.py` | `Merged` | Core energy filtering is ported, but native-Python real-volume parity still depends on octave/downsampling details |
| `construct_structuring_element.m` | `source/slavv/core/energy.py` | `Merged` | Folded into the Python energy module |
| `construct_structuring_elements.m` | `source/slavv/core/energy.py` | `Merged` | |
| `calculate_linear_strel.m` | `source/slavv/core/energy.py` | `Merged` | |
| `calculate_linear_strel_range.m` | `source/slavv/core/energy.py` | `Merged` | |
| `get_energy_V200.m` | `source/slavv/core/energy.py` | `Merged` | |
| `h52mat.m` | `source/slavv/io/matlab_bridge.py` | `Merged` | MATLAB HDF5 sidecars are now imported into pipeline-compatible checkpoints |

## Vertex Detection

| MATLAB File | Python Location | Status | Notes |
| --- | --- | --- | --- |
| `choose_vertices_V200.m` | `source/slavv/core/tracing.py` | `Merged` | MATLAB-style crop/paint logic is ported |
| `vertex_info_extractor.m` | `source/slavv/io/matlab_parser.py` | `Ported` | |
| `vertex_feature_extractor.m` | `source/slavv/analysis/ml_curator.py` | `Merged` | |
| `uncuratedInfoExtractor.m` | `source/slavv/analysis/ml_curator.py` | `Ported` | |
| `vertex_curator.m` | `source/slavv/visualization/interactive_curator.py` | `Ported` | Interactive curation UI |

## Edge Extraction And Tracing

| MATLAB File | Python Location | Status | Notes |
| --- | --- | --- | --- |
| `get_edges_V200.m` | `source/slavv/core/tracing.py` | `Merged` | Main edge extraction flow and candidate cleanup |
| `get_edges_V204.m` | `source/slavv/core/tracing.py` | `Merged` | MATLAB frontier semantics inform the parity-only tracer path |
| `get_edges_for_vertex.m` | `source/slavv/core/tracing.py` | `Merged` | Ported as the parity-only best-first frontier search for MATLAB-energy runs |
| `choose_edges_V200.m` | `source/slavv/core/tracing.py` | `Merged` | Duplicate cleanup, conflict handling, and graph-facing edge selection |
| `add_vertices_to_edges.m` | `source/slavv/core/tracing.py` | `Merged` | |
| `clean_edges.m` | `source/slavv/core/tracing.py` | `Merged` | |
| `clean_edges_cycles.m` | `source/slavv/core/tracing.py` | `Merged` | |
| `clean_edges_hairs.m` | `source/slavv/core/tracing.py` | `Merged` | |
| `clean_edges_orphans.m` | `source/slavv/core/tracing.py` | `Merged` | |
| `clean_edges_vertex_degree_excess.m` | `source/slavv/core/tracing.py` | `Merged` | |
| `clean_edge_pairs.m` | `source/slavv/core/tracing.py` | `Merged` | |
| `edge_info_extractor.m` | `source/slavv/io/matlab_parser.py` | `Ported` | |
| `edge_curator.m` | `source/slavv/visualization/interactive_curator.py` | `Ported` | Interactive curation UI |

## Network And Graph Construction

| MATLAB File | Python Location | Status | Notes |
| --- | --- | --- | --- |
| `get_network_V200.m` | `source/slavv/core/graph.py` | `Ported` | |
| `combine_strands.m` | `source/slavv/core/graph.py` | `Merged` | |
| `fix_strand_vertex_mismatch.m` | `source/slavv/core/graph.py` | `Merged` | |
| `export_strand_data.m` | `source/slavv/io/exporters.py` | `Ported` | |
| `calculate_network_statistics.m` | `source/slavv/analysis/geometry.py` | `Ported` | |

## File I/O And Format Conversion

| MATLAB File | Python Location | Status | Notes |
| --- | --- | --- | --- |
| `dicom2tif.m` | `source/slavv/io/tiff.py` | `Ported` | |
| `tif2mat.m` | `source/slavv/io/tiff.py` | `Ported` | |
| `casX2mat.m` | `source/slavv/io/network_io.py` | `Ported` | |
| `casx_mat2file.m` | `source/slavv/io/network_io.py` | `Ported` | |
| `casx2vmv.m` | `source/slavv/io/network_io.py` | `Ported` | |
| `vmv_mat2file.m` | `source/slavv/io/network_io.py` | `Ported` | |
| `registration_txt2mat.m` | `source/slavv/io/exporters.py` | `Ported` | |
| `partition_casx_by_xy_bins.m` | `source/slavv/io/exporters.py` | `Ported` | |

## Visualization, Analysis, And Helpers

| MATLAB File | Python Location | Status | Notes |
| --- | --- | --- | --- |
| `visualize_vertices_V200.m` | `source/slavv/visualization/network_plots.py` | `Ported` | |
| `visualize_edges_V180.m` | `source/slavv/visualization/network_plots.py` | `Merged` | |
| `visualize_strands.m` | `source/slavv/visualization/network_plots.py` | `Merged` | |
| `flow_field_subroutine.m` | `source/slavv/visualization/volume_rasterization.py` | `Ported` | |
| `calculate_image_stats.m` | `source/slavv/analysis/geometry.py` | `Ported` | |
| `calculate_surface_area.m` | `source/slavv/analysis/geometry.py` | `Ported` | |
| `weighted_KStest2.m` | `source/slavv/utils/math.py` | `Ported` | |
| `fourier_transform_V2.m` | `source/slavv/utils/math.py` | `Ported` | |
| `get_vessel_directions_V3.m` | `source/slavv/analysis/geometry.py` | `Merged` | |

## Current Parity Focus

As of March 27, 2026:

- The MATLAB batch importer in `source/slavv/io/matlab_bridge.py` now loads
  real HDF5 energy sidecars into checkpoint-compatible `energy_data` payloads.
- Exact vertex parity is reached when the Python pipeline runs under imported
  MATLAB energy with `comparison_exact_network=True`.
- The parity-only frontier tracer in `source/slavv/core/tracing.py` ports the
  best-first search behavior from `get_edges_for_vertex.m` closely enough to
  remove dangling-path collapse as the dominant failure mode.
- Edge cleanup in `source/slavv/core/tracing.py` now applies MATLAB-shaped
  duplicate ordering for parity runs, including deterministic shorter-trace
  tie-breaking before downstream pruning.
- Network construction in `source/slavv/core/graph.py` now shares a parity-aware
  topology path between fresh and resumable runs and emits additive
  `strands_to_vertices` payloads for exact-network comparisons.
- Fresh live MATLAB comparison runs now confirm exact vertex parity but still
  miss exact edge and strand parity on the canonical parity path.
- See `docs/PARITY_FINDINGS_2026-03-27.md` for the current clean-run numbers,
  blocker analysis, and recommended next-phase investigations.

## Upstream Files Intentionally Not Ported

The upstream checkout still contains many files that are intentionally outside
the scope of the maintained Python package:

- `vectorization_script_*.m` user-specific run scripts
- one-off study and demo scripts such as registration or noise-sensitivity
  experiments
- backups and superseded variants
- ad-hoc utilities that are replaced by the Python CLI or notebook workflows

Run `python workspace/scripts/maintenance/check_mapped.py` from the repository
root if you need a generated appendix of upstream `.m` files that are not
mentioned explicitly in this document.
