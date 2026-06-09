# Handoff Report: R1. Vertices Parity

## Observation
In the `vertices` stage, there is a discrepancy in how candidate vertices are sorted globally before they undergo conflict resolution (`choose_vertices_matlab_style` in Python / `crop_vertices_V200` and subsequent iteration in MATLAB). 

1. In MATLAB (`external/Vectorization-Public/source/get_vertices_V200.m`), vertices are extracted chunk by chunk using a `parfor` loop. Within each chunk, ties for minimum energy are broken by MATLAB's column-major linear index (because `min()` returns the first occurrence).
2. The chunk results are concatenated in chunk index order (`space_subscripts = cell2mat( space_subscripts );`).
3. In `vectorize_V200.m` (line 2892), MATLAB performs a global sort: `[ vertex_energies, sorted_indices ] = sort( vertex_energies );`. Because MATLAB's `sort` is stable, energy ties across the entire volume retain their concatenated order (i.e., by `chunk_index`, then by chunk-local linear index).
4. In Python (`slavv_python/processing/stages/vertices/manager.py`), candidates are correctly gathered from chunks in chunk-index order. However, the Python global sort calls `sort_vertex_order(..., context["image_shape"], ...)` with the **global image shape**.
5. Python's `sort_vertex_order` uses `np.lexsort((linear_indices, vertex_energies))`. This explicitly breaks global energy ties using the **global linear index**.
6. Global linear index order is not equivalent to chunk-index order (because chunks overlap and divide the volume non-linearly).

## Logic Chain
1. Python's `matlab_vertex_candidates` correctly replicates the chunk-by-chunk extraction and concatenates the results in the exact same chunk order as MATLAB.
2. Inside a chunk, `sort_vertex_order` correctly uses the chunk's shape to break ties by chunk-local linear index, matching MATLAB's `min()` behavior.
3. However, calling `sort_vertex_order` globally re-sorts the entire candidate list by global linear indices. 
4. Because the global sort determines the evaluation order for `choose_vertices_matlab_style` (the conflict/painting resolution), any tie between overlapping voxels in different chunks will be resolved by global linear index in Python, but by chunk index in MATLAB.
5. This difference in evaluation order causes Python to select a different winning vertex in overlapping border regions, leading to missing and extra vertices in the final `VertexSet`.

## Caveats
- The canonical energy rerun for `180709_E_crop_M_exact` was still actively running as a detached PID, so the Python `prove-exact` harness for `vertices` could not be executed to completion during this session without waiting for `energy` to recompute. The analysis was conducted statically against the MATLAB `Vectorization-Public` truth code.

## Conclusion
The exact parity failure in the `vertices` stage is caused by Python performing a global tie-breaking sort using global linear indices, whereas MATLAB uses a stable sort that preserves the implicit chunk-index ordering.

To fix this, Python needs two distinct sorting behaviors:
1. **Chunk-local sort**: Continue using `np.lexsort` with linear indices (calculated via chunk shape) inside `matlab_vertex_candidates_in_chunk`.
2. **Global sort**: Use a stable sort on `vertex_energies` only (e.g., `np.argsort(vertex_energies, kind='stable')`) in `VertexManager._run_resumable` and `VertexManager._run_ephemeral`. Do not use `lexsort` with global linear indices for the global sort.

## Verification Method
1. Modify `VertexManager._run_resumable` and `VertexManager._run_ephemeral` to replace the `sort_vertex_order` call with a stable `argsort` on `energies`. If `energy_sign < 0`, sort ascending; if `> 0`, sort descending (using `kind='stable'`).
2. Once the energy rerun completes (PID 25248 or a fresh run), execute:
   `python scripts/cli/parity_experiment.py prove-exact --source-run-root workspace/runs/oracle_180709_E/crop_M_exact --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact --oracle-root workspace/oracles/180709_E_crop_M --stage vertices`
3. The proof should report 0 missing and 0 extra vertices.
