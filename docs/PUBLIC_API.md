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

These APIs are considered stable and will be maintained for external consumers.
