# Bugfix Requirements Document

## Introduction

The v22 global watershed implementation contains a critical pointer corruption bug that causes invalid pointer indices during backtracking. This bug results in 65% match rate with MATLAB (target: 100%), 890 missing candidates (35% of MATLAB's 2533 total), and 477 extra candidates. The bug manifests as pointer indices that are 10-100x larger than valid for their associated scale's lookup table (LUT), causing out-of-range errors during edge tracing.

Through systematic investigation with comprehensive diagnostic logging, the root cause has been identified as a scale mismatch: pointers are created for a clipped scale index but the unclipped scale label is written to the size_map. During backtracking, the unclipped scale is read from size_map and used to build the wrong LUT, causing pointer indices to be out of range.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN `current_scale_label` exceeds the valid range `[1, len(lumen_radius_microns)]` THEN the system clips the scale index to `[0, len(lumen_radius_microns) - 1]` for LUT building but writes the unclipped `current_scale_label` to `size_map`

1.2 WHEN backtracking reads `tracing_scale_label` from `size_map` at a location THEN the system uses the unclipped scale value to build a LUT that does not match the LUT used during pointer creation

1.3 WHEN the backtracking LUT is smaller than the pointer creation LUT THEN the system encounters pointer indices that are out of range (e.g., pointer 1373 for scale 6 with LUT size 81)

1.4 WHEN pointer indices are out of range during backtracking THEN the system logs errors and breaks the trace, resulting in missing edge candidates

1.5 WHEN the scale mismatch occurs across multiple locations THEN the system produces 890 missing candidates (35% of expected 2533) and 477 spurious extra candidates

### Expected Behavior (Correct)

2.1 WHEN creating pointers for a location with `current_scale_label` THEN the system SHALL clip the scale index and use the clipped value for both LUT building and `size_map` storage

2.2 WHEN backtracking reads `tracing_scale_label` from `size_map` THEN the system SHALL build a LUT that exactly matches the LUT used during pointer creation for that location

2.3 WHEN backtracking uses a pointer index to traverse THEN the system SHALL find the pointer index within the valid range `[1, len(linear_offsets)]` for the scale's LUT

2.4 WHEN all pointers are valid during backtracking THEN the system SHALL successfully trace all edge candidates without out-of-range errors

2.5 WHEN the scale consistency is maintained throughout pointer lifecycle THEN the system SHALL achieve 100% match rate with MATLAB (2533 candidates) with zero missing or extra candidates

### Unchanged Behavior (Regression Prevention)

3.1 WHEN a location has `current_scale_label` within the valid range `[1, len(lumen_radius_microns)]` THEN the system SHALL CONTINUE TO use the scale value without modification

3.2 WHEN creating pointers for in-bounds strel coordinates THEN the system SHALL CONTINUE TO generate 1-based pointer indices into the LUT

3.3 WHEN writing pointers to unclaimed locations (vertex_index == 0) THEN the system SHALL CONTINUE TO respect the `is_without_vertex` constraint and not overwrite existing pointers

3.4 WHEN backtracking from a location with pointer value 0 THEN the system SHALL CONTINUE TO terminate the trace at that origin location

3.5 WHEN building LUTs for different scale indices THEN the system SHALL CONTINUE TO produce scale-specific LUT sizes based on `lumen_radius_microns` and `step_size_per_origin_radius`

3.6 WHEN tracing edge halves and finalizing edge traces THEN the system SHALL CONTINUE TO sample energy and scale values from the correct MATLAB-order linear indices

3.7 WHEN calculating edge metrics from energy traces THEN the system SHALL CONTINUE TO use the existing metric calculation logic

3.8 WHEN filtering strel coordinates for boundary conditions THEN the system SHALL CONTINUE TO apply the existing valid_mask logic for coordinate bounds checking
