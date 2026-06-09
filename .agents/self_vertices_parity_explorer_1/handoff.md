# Handoff: Vertices Parity Analysis

## Observation
When running the MATLAB exact parity proof for the `vertices` stage on the canonical or crop volumes, Python fails to perfectly match MATLAB's output. The user prompt explicitly calls out "Lowest Linear Index Priority" as a tie-breaking rule for vertices with identical energies.

Reviewing MATLAB's `Vectorization-Public/source/get_vertices_V200.m`:
1. Chunks are processed using `parfor chunk_index = chunk_index_range` where `chunk_index_range` is `1 : number_of_chunks`.
2. `ind2sub( chunk_lattice_dimensions, chunk_index )` maps the 1D chunk index to 3D chunk coordinates. In MATLAB, `ind2sub` varies the first dimension (Y) fastest.
3. The results from the parallel workers are concatenated using `space_subscripts = cell2mat( space_subscripts );`. This vertically concatenates the results in the strict `chunk_index` order (Y-fastest, then X, then Z).
4. Later, in `vectorize_V200.m`, the concatenated vertex array is sorted purely by energy: `[ vertex_energies, sorted_indices ] = sort( vertex_energies );`. MATLAB's `sort` is stable, preserving the original chunk-concatenation order for vertices with identical energies.

Reviewing Python's `slavv_python/processing/stages/vertices/detection.py`:
1. `iter_overlapping_chunks` generates chunks using nested loops: `for y_index ...: for x_index ...: for z_index ...:`. The innermost loop is Z, meaning Z varies fastest.
2. `matlab_vertex_candidates` collects parallel results in the generator's Z-fastest order.
3. In `manager.py`, Python performs a stable sort on the energies: `np.argsort(vertex_energies, kind="stable")`, matching MATLAB's global sort.

## Logic Chain
1. Both MATLAB and Python handle intra-chunk tie-breaking identically (MATLAB via `min()` in the chunk, Python via `lexsort` on chunk-relative linear indices inside `matlab_vertex_candidates_in_chunk`).
2. Both MATLAB and Python use a stable sort globally across all vertices, meaning inter-chunk tie-breaking falls back entirely to the original array order prior to sorting.
3. Because Python yields chunks from `iter_overlapping_chunks` in Z-fastest order, and MATLAB iterates `chunk_index` in Y-fastest order, their pre-sort arrays are ordered differently.
4. When two vertices with identical energies reside in different chunks, the stable sort preserves their chunk-concatenation order. Since the concatenation orders differ, Python will prioritize a different vertex than MATLAB when breaking inter-chunk ties, leading to downstream divergences in `choose_vertices_matlab_style` where one vertex suppresses another based on traversal order.

## Caveats
- This assumes that `energy_sign` is `< 0` for MATLAB parity workflows (which it is, as MATLAB relies on min-projections).
- This assumes that `joblib.Parallel` returns results in the same order as the input generator, which it guarantees by default.
- We did not manually implement the fix, per instruction. The proposed fix is to reverse the loop nesting in `iter_overlapping_chunks` so that `y_index` is the innermost loop and `z_index` is the outermost.

## Conclusion
The exact parity failure in the `vertices` stage stems from a mismatched chunk traversal order. Python concatenates chunk results in Z-fastest order (due to nested loop structure in `iter_overlapping_chunks`), while MATLAB concatenates them in Y-fastest order (due to `ind2sub` behavior over `1:number_of_chunks`). Because the global vertex sort is stable, this difference alters the tie-breaking sequence for identical-energy vertices across chunks, diverging the downstream exclusion logic. 

**Fix Strategy:** Reorder the nested loops in `slavv_python/processing/stages/vertices/detection.py::iter_overlapping_chunks` to make `y_index` the innermost loop and `z_index` the outermost loop, ensuring chunks are yielded (and thus concatenated) in Y-fastest order.

## Verification Method
1. Implement the loop reordering in `slavv_python/processing/stages/vertices/detection.py`.
2. Run the exact parity harness for the vertices stage:
   `python scripts/cli/parity_experiment.py prove-exact --source-run-root workspace/runs/oracle_180709_E/crop_M_exact --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact --oracle-root workspace/oracles/180709_E_crop_M --stage vertices`
3. The harness should report 0 missing and 0 extra vertices.
