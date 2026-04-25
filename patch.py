import sys
content = open('source/core/_edge_candidates/global_watershed.py').read()

# Fix 1: Cycle detector
old1 = """def _matlab_global_watershed_trace_half(
    start_linear: int,
    *,
    pointer_map: np.ndarray,
    size_map: np.ndarray,
    shape: tuple[int, int, int],
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    step_size_per_origin_radius: float,
) -> list[int]:
    \"\"\"Trace one MATLAB watershed half-edge back to its zero-pointer origin.\"\"\"
    traced: list[int] = []
    tracing_linear = int(start_linear)
    while True:
        traced.append(int(tracing_linear))"""
new1 = """def _matlab_global_watershed_trace_half(
    start_linear: int,
    *,
    pointer_map: np.ndarray,
    size_map: np.ndarray,
    shape: tuple[int, int, int],
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    step_size_per_origin_radius: float,
) -> list[int]:
    \"\"\"Trace one MATLAB watershed half-edge back to its zero-pointer origin.\"\"\"
    traced: list[int] = []
    visited: set[int] = set()
    tracing_linear = int(start_linear)
    while True:
        if tracing_linear in visited:
            import logging
            logging.error(f"Cycle detected in global watershed backtrack at linear index {tracing_linear}. Breaking to prevent infinite loop.")
            break
        visited.add(tracing_linear)
        traced.append(int(tracing_linear))"""
if old1 in content:
    content = content.replace(old1, new1)
    print("Fix 1 applied")

# Fix 2: 1D flat volume optimization
old2 = """def _sample_volume_from_matlab_linear_trace(
    linear_trace: list[int],
    volume: np.ndarray,
) -> np.ndarray:
    \"\"\"Sample one volume exactly at normalized MATLAB-order linear indices.\"\"\"
    if not linear_trace:
        return np.zeros((0,), dtype=np.asarray(volume).dtype)
    flat_volume = np.asarray(volume).ravel(order="F")"""
new2 = """def _sample_volume_from_matlab_linear_trace(
    linear_trace: list[int],
    volume: np.ndarray,
) -> np.ndarray:
    \"\"\"Sample one volume exactly at normalized MATLAB-order linear indices.\"\"\"
    if not linear_trace:
        return np.zeros((0,), dtype=np.asarray(volume).dtype)
    if np.asarray(volume).ndim == 1:
        flat_volume = np.asarray(volume)
    else:
        flat_volume = np.asarray(volume).ravel(order="F")"""
if old2 in content:
    content = content.replace(old2, new2)
    print("Fix 2 applied")

# Fix 3: Vectorized strel length lookup
old3 = """    scale_indices = np.clip(scale_labels - 1, 0, len(lumen_radius_microns) - 1)
    strel_lengths = np.asarray(
        [
            len(
                _build_matlab_global_watershed_lut(
                    int(scale_index),
                    size_of_image=pointer_map.shape,
                    lumen_radius_microns=lumen_radius_microns,
                    microns_per_voxel=microns_per_voxel,
                    step_size_per_origin_radius=step_size_per_origin_radius,
                )["linear_offsets"]
            )
            for scale_index in scale_indices
        ],
        dtype=np.float32,
    )
    scaled_pointer_map[pointer_mask] = ("""
new3 = """    scale_indices = np.clip(scale_labels - 1, 0, len(lumen_radius_microns) - 1)
    unique_lengths = np.zeros(len(lumen_radius_microns), dtype=np.float32)
    for i in range(len(lumen_radius_microns)):
        unique_lengths[i] = len(
            _build_matlab_global_watershed_lut(
                i,
                size_of_image=pointer_map.shape,
                lumen_radius_microns=lumen_radius_microns,
                microns_per_voxel=microns_per_voxel,
                step_size_per_origin_radius=step_size_per_origin_radius,
            )["linear_offsets"]
        )
    strel_lengths = unique_lengths[scale_indices]
    scaled_pointer_map[pointer_mask] = ("""
if old3 in content:
    content = content.replace(old3, new3)
    print("Fix 3 applied")

# Fix 4: Pre-flatten volume maps before trace gathering
old4 = """    for (start_vertex_index, end_vertex_index), (half_1, half_2) in zip(edge_pairs, edge_halves):
        if end_vertex_index == number_of_vertices + 1:
            continue
        trace, energy_trace, scale_trace = _matlab_global_watershed_finalize_edge_trace(
            half_1,
            half_2,
            shape=shape,
            energy_map=energy_map,
            scale_image=original_scale_image,
        )"""

new4 = """    flat_energy_map = np.asarray(energy_map, dtype=np.float32).ravel(order="F")
    if original_scale_image is not None:
        flat_scale_image = np.asarray(original_scale_image, dtype=np.int16).ravel(order="F")
    else:
        flat_scale_image = None

    for (start_vertex_index, end_vertex_index), (half_1, half_2) in zip(edge_pairs, edge_halves):
        if end_vertex_index == number_of_vertices + 1:
            continue
        trace, energy_trace, scale_trace = _matlab_global_watershed_finalize_edge_trace(
            half_1,
            half_2,
            shape=shape,
            energy_map=flat_energy_map,
            scale_image=flat_scale_image,
        )"""
if old4 in content:
    content = content.replace(old4, new4)
    print("Fix 4 applied")

open('source/core/_edge_candidates/global_watershed.py', 'w').write(content)
