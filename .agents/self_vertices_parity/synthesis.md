## Consensus
Explorer 2 identified that the exact parity failure in the `vertices` stage is caused by Python performing a global tie-breaking sort using global linear indices (`np.lexsort((linear_indices, vertex_energies))`), whereas MATLAB uses a stable sort on energies only (`sort(vertex_energies)`) which preserves the implicit chunk-index ordering.

## Resolved Conflicts
None. 

## Dissenting Views
None. Explorer 2 provided a complete logical chain for the discrepancy based on the source code.

## Gaps
None. 

## Fix Strategy
Modify `VertexManager._run_resumable` and `VertexManager._run_ephemeral` (in `slavv_python/processing/stages/vertices/manager.py`).
1. Replace the `sort_vertex_order` call with a stable `argsort` on `energies`.
   - `energy_sign < 0` -> sort ascending
   - `energy_sign > 0` -> sort descending
   - Use `kind='stable'` to preserve the chunk order.
2. Ensure we do not use `lexsort` with global linear indices for the global sort. The chunk-local sort in `matlab_vertex_candidates_in_chunk` already handles the linear index tie-breaking correctly.
