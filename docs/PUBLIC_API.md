# Public API

This document outlines the stable entry points exposed by the Python port of SLAVV.

## [SLAVVProcessor](../slavv-streamlit/src/vectorization_core.py)

- `process_image(image, parameters)`: run the full SLAVV pipeline and return energy, vertex, edge, and network data.
- `calculate_energy_field(image, params)`: compute the multi-scale energy field.
- `extract_vertices(energy_data, params)`: detect vessel vertices as local extrema.
- `extract_edges(energy_data, vertices, params)`: trace edges between vertices using gradient descent.
- `extract_edges_watershed(energy_data, vertices, params)`: alternative edge extraction via watershed regions.
- `construct_network(edges, vertices, params)`: build strands and adjacency information for the final network.

## Utility Functions

- [`preprocess_image`](../slavv-streamlit/src/vectorization_core.py): normalize intensities and optionally remove axial banding.
- [`validate_parameters`](../slavv-streamlit/src/vectorization_core.py): populate defaults and sanity‑check processing parameters.
- [`get_chunking_lattice`](../slavv-streamlit/src/vectorization_core.py): generate overlapping tile slices for chunked energy-field processing.
- [`calculate_network_statistics`](../slavv-streamlit/src/vectorization_core.py): compute strand counts, lengths, and other network metrics.
- [`calculate_surface_area`](../slavv-streamlit/src/vectorization_core.py): estimate total vessel surface area.
- [`crop_vertices`](../slavv-streamlit/src/vectorization_core.py): filter vertices within an axis-aligned bounding box.
- [`crop_edges`](../slavv-streamlit/src/vectorization_core.py): drop edges whose endpoints fall outside a vertex mask.
- [`crop_vertices_by_mask`](../slavv-streamlit/src/vectorization_core.py): retain vertices located inside a binary mask volume.
- [`load_network_from_mat`](../slavv-streamlit/src/io_utils.py): read vertices, edges, and radii from MATLAB `.mat` files.
- [`load_network_from_casx`](../slavv-streamlit/src/io_utils.py): read network data from CASX XML files.
- [`load_network_from_vmv`](../slavv-streamlit/src/io_utils.py): read network data from VMV text files.
- [`load_tiff_volume`](../slavv-streamlit/src/io_utils.py): read a 3D grayscale TIFF volume with validation.
- [`extract_uncurated_info`](../slavv-streamlit/src/ml_curator.py): derive vertex and edge feature arrays without curation.

These APIs are considered stable and will be maintained for external consumers.
