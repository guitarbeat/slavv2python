# Handoff Report: R1. Vertices Parity

## 1. Observation
- `prove-exact` could not complete immediately because an `energy` run was currently active on the `crop_M_exact` run directory, preventing immediate vertex comparison.
- Code inspection reveals that Python's `sort_vertex_order` (in `slavv_python/processing/stages/vertices/results.py`) uses `np.lexsort((linear_indices, vertex_energies))` to break ties globally using MATLAB-style column-major linear indices.
- Python uses `sort_vertex_order` twice: once inside `matlab_vertex_candidates_in_chunk` to sort candidates within a chunk, and again globally in `manager.py` (after cropping) to sort the entire volume's candidates.
- In MATLAB, `get_vertices_V200.m` finds candidates chunk by chunk. The `min()` function inherently prioritizes the lowest column-major linear index for ties *within* a chunk. Chunks are then concatenated with `cell2mat` in chunk-index order.
- Before choosing vertices, MATLAB's `vectorize_V200.m` sorts the concatenated candidates using `[ vertex_energies, sorted_indices ] = sort( vertex_energies );`. MATLAB's `sort` is **stable** and therefore uses the *input order* (i.e., chunk-concatenation order) as the global tie-breaker, not the global linear index.

## 2. Logic Chain
- MATLAB implements "Lowest Linear Index Priority" locally within each chunk because it relies on the behavior of `min(energy_chunk_temp(:))`.
- When MATLAB concatenates the chunks, the relative order of vertices with identical energies from *different* chunks is governed purely by chunk iteration order (y, x, z block order).
- Because a 3D chunking lattice creates blocks of indices, it is possible for an earlier chunk to contain higher global linear indices than a subsequent chunk. Therefore, chunk-order tie-breaking diverges from strict global linear-index tie-breaking.
- By applying `np.lexsort((linear_indices, vertex_energies))` globally in `manager.py`, Python destroys the chunk-concatenation order and forces a strict global linear-index priority.
- This creates a structural divergence from MATLAB's 1:1 behavior. To achieve exact parity, Python must break ties within chunks using linear indices, but must break ties globally using a stable sort that preserves the chunk concatenation order.

## 3. Caveats
- I did not wait for the 4-hour `parity_experiment.py` energy stage run to finish, so I did not observe the exact failure trace directly via the parity harness. The investigation was purely static code analysis based on the domain definitions of "Lowest Linear Index Priority" and sorting semantics.
- I assumed MATLAB's `cell2mat` on a `parfor` output preserves the `1 : number_of_chunks` order, which aligns with standard MATLAB parallel pool behavior.

## 4. Conclusion
The `vertices` exact parity discrepancy is caused by Python incorrectly applying global linear-index tie-breaking during the final sorting step in `manager.py`. MATLAB uses a stable sort that preserves the chunk-concatenation order for global ties. 

**Fix Strategy:**
Separate the tie-breaking logic. Retain linear-index tie-breaking inside the chunk processing step (`matlab_vertex_candidates_in_chunk`), but use a strict stable sort (e.g., `np.argsort(kind='stable')`) on energy alone for the global sort in `manager.py`. One approach is to add a `stable_only=True` parameter to `sort_vertex_order` and invoke it differently depending on the context.

## 5. Verification Method
1. Implement the fix to ensure the global sort in `manager.py` stably sorts by energy only.
2. Run the `prove-exact` pre-gate command once the energy run has completed:
   `python scripts/parity_experiment.py prove-exact --source-run-root workspace/runs/oracle_180709_E/crop_M_exact --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact --oracle-root workspace/oracles/180709_E_crop_M --stage vertices`
3. The exact proof must report exactly 0 missing and 0 extra vertices.
