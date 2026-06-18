---
title: MATLAB Stride Phase Lead (Energy Spatial Shift)
module: processing/energy
tags: [parity, coordinate-alignment, stride-phase, matlab]
problem_type: parity
resolution_type: code_fix
---

# MATLAB Stride Phase Lead (Energy Spatial Shift)

## Problem
During Energy stage parity certification for the `180709_E` volume, a significant spatial mismatch (aリード of ~15 pixels) was observed between the Python energy map and the MATLAB oracle. Even with orientation aligned to `[Y, X, Z]`, the values at the origin did not match, and the entire structure appeared shifted along the X and Y axes.

## Evidence
`slavv parity prove-exact` reported massive value mismatches at the origin. Analytical inspection of the coarse-grid interpolation meshes showed that Python was starting at index `0`, while the MATLAB oracle lead suggested it was starting at a non-zero offset. 

## Root Cause
MATLAB's `get_starts_and_counts_V200` utility, used in the vesselness engine, employs a "Last Chunk Alignment" rule. It calculates the reading start such that the strided read ends exactly on the last pixel of the volume:
`reading_count = 1 + rf * floor((size - 1) / rf)`
`reading_start = reading_end - reading_count + 1`

For a single-chunk volume (like a small crop volume at a high resolution factor `rf`), this results in a lead offset of `(size - 1) % rf`. For `rf=9` and `size=256`, the lead is `255 % 9 = 3`. For `rf=20`, the lead is `255 % 20 = 15`.

## Solution
Updated `_downsample_volume` in `hessian_response.py` to calculate and apply this stride phase lead when `comparison_exact_network` is enabled.

```python
if alignment == "matlab":
    # MATLAB shifts the lead to align the final pixel
    starts = [(image.shape[axis] - 1) % factors[axis] for axis in range(3)]
else:
    starts = [0, 0, 0]
```

## Verification
Rerunning `crop_M_exact` showed that the origin energy value matched the MATLAB oracle (`-20.3757...`) to within 14 decimal places, effectively resolving the spatial shift blocker.

## Follow-Up
Ensure that any future "Standard" vs "Exact" unification preserves this alignment rule as a configurable policy to avoid breaking standard-path regression tests.
