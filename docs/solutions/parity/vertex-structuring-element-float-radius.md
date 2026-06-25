---
title: Vertex NMS structuring element must use float radii (MATLAB construct_structuring_element)
module: pipeline/vertices
tags: [vertices, structuring-element, ellipsoid, non-max-suppression, parity]
problem_type: parity
resolution_type: code_fix
---

# Vertex NMS structuring element must use float radii

## Problem
`prove-exact --stage vertices` vs `180709_E_crop_M_v2` failed with a structural
divergence: Python found 12,696 vertices vs MATLAB's 13,706 (1,216 missing + 206
extra). Energy was already certified and vertex *candidate detection* matched, so
the gap was in the `choose_vertices` non-maximum-suppression step.

## Evidence
- Set diff of vertex positions: 12,490 shared, 1,216 MATLAB-only, 206 Python-only
  (both missing AND extra → a suppression-volume sizing difference, not a threshold).
- `ellipsoid_offsets` voxel counts vs MATLAB `construct_structuring_element.m` for
  anisotropic radii diverged in both directions (e.g. radii `[3.4,3.4,8.5]`:
  MATLAB 429 vs Python 335; `[2.8,2.8,7.0]`: 227 vs 259).

## Root Cause
`slavv_python/pipeline/vertices/detection.py::ellipsoid_offsets` rounded the radii
to integers and called `skimage.draw.ellipsoid`. MATLAB
`construct_structuring_element.m` uses `round(radii)` only for the grid
*dimensions/center* but the **unrounded float radii** in the membership test
`(dy/r0)^2 + (dx/r1)^2 + (dz/r2)^2 <= 1`. The rounded/skimage SE has a different
voxel set at fractional radii, so NMS over- or under-suppressed neighbors.

## Solution
Port `construct_structuring_element.m` faithfully in `ellipsoid_offsets`: grid dims
and center from `round(radii)`, membership from float radii. Drop the skimage call.

## Verification
`resume-exact-run --force-rerun-from vertices --stop-after vertices` then
`prove-exact --stage vertices` vs `180709_E_crop_M_v2`: positions + scales match
**exactly** (13,706 = 13,706; 0 missing/extra). Regression test
`test_ellipsoid_offsets_matches_matlab_structuring_element` asserts the MATLAB
voxel counts (19/57/135/227/429 + isotropic 7/33/123). Commit `2215b059`.

## Follow-Up
The same float-radius SE pattern applies anywhere a scale-dependent ellipsoid is
built; reuse `ellipsoid_offsets`, do not reach for `skimage.draw.ellipsoid`.
