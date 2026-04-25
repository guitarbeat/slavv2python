import sys
content = open('source/core/_edge_candidates/global_watershed.py').read()

old1 = """    traced: list[int] = []
    tracing_linear = int(start_linear)
    while True:
        traced.append(int(tracing_linear))"""
new1 = """    traced: list[int] = []
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

old2 = """    if not linear_trace:
        return np.zeros((0,), dtype=np.asarray(volume).dtype)
    flat_volume = np.asarray(volume).ravel(order="F")"""
new2 = """    if not linear_trace:
        return np.zeros((0,), dtype=np.asarray(volume).dtype)
    if np.asarray(volume).ndim == 1:
        flat_volume = np.asarray(volume)
    else:
        flat_volume = np.asarray(volume).ravel(order="F")"""
if old2 in content:
    content = content.replace(old2, new2)
    print("Fix 2 applied")

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
