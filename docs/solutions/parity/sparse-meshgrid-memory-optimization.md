---
title: Sparse Meshgrid Memory Optimization
module: processing/energy
tags: [memory, performance, exact-route, meshgrid]
problem_type: parity
resolution_type: code_fix
---

# Sparse Meshgrid Memory Optimization

## Problem
During the exact-route Energy stage processing, canonical volumes (512x512x64) were experiencing severe memory overhead when computing `_interp3_matlab_linear_inf` (which lives in `slavv_python/pipeline/energy/matlab_get_energy_v202_chunked.py`). The pipeline was generating fully dense 4D arrays `(3, Y, X, Z)` for the coordinate grid to pass into interpolation, which consumed >400MB of RAM per chunk and risked `ArrayMemoryError` on constrained developer machines.

## Evidence
Memory profilers showed massive allocations occurring at the `np.meshgrid(..., indexing="ij")` and `np.stack` step. The `np.meshgrid(..., sparse=True)` change applies to `_upsample_volume` in `slavv_python/pipeline/energy/matlab_energy_filter_v200.py`, distinct from `_interp3_matlab_linear_inf` in `matlab_get_energy_v202_chunked.py`.

## Root Cause
The legacy MATLAB-compatible `_interp3_matlab_linear_inf` shim was originally written to expect dense coordinate grids because it computed fractional offsets for all coordinates simultaneously before iterating over the interpolation corners. This eagerness was unnecessary since the interpolation logic itself can operate on broadcasted vectors.

## Solution
Refactored `_interp3_matlab_linear_inf` to accept a tuple of sparse 1D arrays and utilized `np.broadcast_to()` on-the-fly inside the valid-corner evaluation loop. Updated the meshgrid creation to use `sparse=True`.
1. Changed `np.meshgrid(..., indexing="ij")` to `np.meshgrid(..., indexing="ij", sparse=True)` in `_upsample_volume` (`matlab_energy_filter_v200.py`).
2. Passed the tuple directly to the interpolator.
3. Updated the interpolation engine to broadcast coordinates dynamically only where valid mask weights applied.

## Verification
```powershell
python -m pytest tests/unit/pipeline/energy/ -v
```
All tests passed (13/13). Memory footprint dropped by >400MB per chunk without any floating-point or parity drift on the exact endpoints.

## Follow-Up
Continue monitoring the canonical tier-3 gate rerun to verify it successfully navigates the energy stage without OOM crashes.