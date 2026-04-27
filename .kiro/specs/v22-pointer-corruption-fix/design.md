# v22 Pointer Corruption Bugfix Design

## Overview

The v22 global watershed implementation has a critical scale mismatch bug causing invalid pointer indices during backtracking. Pointers are created using a clipped scale index but the unclipped scale label is written to size_map. During backtracking, the unclipped scale is read and used to build a mismatched LUT, causing pointer indices to be 10-100x out of range. This results in 65% match rate vs. MATLAB's target 100%, with 890 missing candidates and 477 extra candidates.

The fix is straightforward: return the clipped scale index from `_matlab_global_watershed_current_strel` and use it consistently for both pointer creation and size_map writing. This ensures the scale stored in size_map always matches the scale used to create the pointers, allowing backtracking to reconstruct the correct LUT.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug - when `current_scale_label` exceeds valid range and gets clipped for LUT building but not for size_map storage
- **Property (P)**: The desired behavior - pointers must be valid for the LUT reconstructed during backtracking
- **Preservation**: All existing pointer creation, writing, and backtracking logic that works correctly for in-range scales
- **LUT (Lookup Table)**: Scale-specific array of linear offsets and metadata built by `_build_matlab_global_watershed_lut`
- **Pointer Index**: 1-based index into a scale's LUT, stored in pointer_map to enable backtracking
- **Scale Clipping**: `clip(scale_label - 1, 0, len(lumen_radius_microns) - 1)` applied to ensure valid array indexing
- **size_map**: 3D array storing the scale label for each location, used during backtracking to reconstruct the correct LUT
- **pointer_map**: 3D array storing pointer indices for backtracking from claimed locations to their origins

## Bug Details

### Bug Condition

The bug manifests when `current_scale_label` exceeds the valid range `[1, len(lumen_radius_microns)]`. The `_matlab_global_watershed_current_strel` function clips the scale for LUT building but returns the original unclipped label. The main loop then writes this unclipped label to size_map, creating a mismatch between the scale used to create pointers and the scale stored for backtracking.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type (current_scale_label: int, lumen_radius_microns: array)
  OUTPUT: boolean
  
  RETURN (input.current_scale_label < 1 OR input.current_scale_label > len(input.lumen_radius_microns))
         AND pointer_created_with_clipped_scale(input)
         AND size_map_written_with_unclipped_scale(input)
END FUNCTION
```

### Examples

- **Example 1**: `current_scale_label = 12`, `len(lumen_radius_microns) = 11`
  - Clipped index: `clip(12 - 1, 0, 10) = 10`
  - LUT built for scale 10 (size ~81 for typical parameters)
  - Pointer created: e.g., 45 (valid for scale 10's LUT)
  - size_map written: 12 (unclipped)
  - During backtracking: reads scale 12, clips to 10, but if scale 12 had been valid it would have a much larger LUT (~200+)
  - Result: Pointer 45 is valid, but the logic is inconsistent

- **Example 2**: `current_scale_label = 6`, pointer created for scale 5 (clipped from 6)
  - Scale 5 LUT size: 81
  - Pointer created: 1373 (intended for a larger scale's LUT)
  - size_map written: 6
  - During backtracking: reads scale 6, clips to 5, builds LUT size 81
  - Result: Pointer 1373 is 17x out of range [1, 81] → ERROR

- **Example 3**: `current_scale_label = 1`, `len(lumen_radius_microns) = 11`
  - Clipped index: `clip(1 - 1, 0, 10) = 0`
  - LUT built for scale 0 (size ~27)
  - Pointer created: 1547 (out of range for scale 0)
  - Result: Pointer corruption

- **Edge Case**: `current_scale_label = 0` (invalid, should be 1-based)
  - Clipped index: `clip(0 - 1, 0, 10) = 0`
  - System should handle gracefully by using clipped scale consistently

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Pointer creation logic for in-range scales must continue to work exactly as before
- LUT building for each scale index must produce identical results
- Backtracking from pointer value 0 must continue to terminate traces correctly
- The `is_without_vertex` constraint must continue to prevent pointer overwrites
- Boundary checking for strel coordinates must remain unchanged
- Energy adjustment and frontier propagation logic must remain unchanged
- Edge metric calculation from energy traces must remain unchanged
- MATLAB-order linear indexing must remain unchanged

**Scope:**
All inputs where `current_scale_label` is within the valid range `[1, len(lumen_radius_microns)]` should be completely unaffected by this fix. This includes:
- Normal pointer creation and storage for in-range scales
- Backtracking with correctly matched scale/pointer pairs
- All other watershed algorithm logic (energy tolerance, branch order, adjacency tracking)

## Hypothesized Root Cause

Based on the investigation handover and diagnostic logging, the root cause is confirmed:

1. **Scale Clipping Inconsistency**: In `_matlab_global_watershed_current_strel` (line ~115), the function clips `current_scale_label` to create `current_scale_index` for LUT building, but does not return the clipped scale. The function implicitly uses the original `current_scale_label` throughout.

2. **Unclipped Scale Storage**: In the main loop (line ~560), the code writes `current_scale_label` (unclipped) to `size_map_flat` via the reveal function (line ~265).

3. **Mismatched LUT Reconstruction**: During backtracking in `_matlab_global_watershed_trace_half` (line ~440), the code reads the unclipped scale from `size_map_flat`, clips it again, and builds a LUT. If the original scale was out of range, this creates a different LUT than the one used during pointer creation.

4. **Pointer Index Overflow**: The pointers were created as indices into the original (clipped) LUT, but during backtracking they're interpreted as indices into a potentially different LUT, causing out-of-range errors.

## Correctness Properties

Property 1: Bug Condition - Scale Consistency for Pointer Validity

_For any_ input where `current_scale_label` is outside the valid range `[1, len(lumen_radius_microns)]`, the fixed function SHALL clip the scale index and use the clipped value consistently for both LUT building during pointer creation AND for writing to size_map, ensuring that backtracking reconstructs the identical LUT and all pointer indices remain within valid range `[1, len(linear_offsets)]`.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

Property 2: Preservation - In-Range Scale Behavior

_For any_ input where `current_scale_label` is within the valid range `[1, len(lumen_radius_microns)]`, the fixed function SHALL produce exactly the same pointer indices, size_map values, and backtracking behavior as the original function, preserving all existing functionality for valid scale inputs.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**

## Fix Implementation

### Changes Required

The fix requires modifying the pointer creation function to return the clipped scale and using that value consistently.

**File**: `source/core/_edge_candidates/global_watershed.py`

**Function**: `_matlab_global_watershed_current_strel` (lines ~100-180)

**Specific Changes**:
1. **Return Clipped Scale**: Add `"scale_label_clipped": current_scale_index + 1` to the return dictionary to provide the clipped scale (converting back to 1-based label)

2. **Update Main Loop**: In `_generate_edge_candidates_matlab_global_watershed` (line ~560), extract the clipped scale from the strel result: `current_scale_label_for_writing = current_strel.get("scale_label_clipped", current_scale_label)`

3. **Pass Clipped Scale to Reveal**: Update the call to `_matlab_global_watershed_reveal_unclaimed_strel` to use `current_scale_label_for_writing` instead of `current_scale_label`

4. **Verify Consistency**: The clipped scale will now be written to size_map, ensuring backtracking reconstructs the correct LUT

5. **Remove Defensive Filtering**: Once the fix is verified, remove the defensive pointer filtering code in the reveal function (lines ~250-260) as it will no longer be needed

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, confirm the bug exists on unfixed code by observing pointer out-of-range errors, then verify the fix eliminates these errors and achieves 100% MATLAB parity.

### Exploratory Bug Condition Checking

**Goal**: Confirm the bug exists on UNFIXED code by observing pointer out-of-range errors and scale mismatches. Verify the root cause hypothesis that unclipped scales are being written to size_map.

**Test Plan**: Run the `capture-candidates` parity experiment on the unfixed code with diagnostic logging enabled. Examine logs for pointer out-of-range errors and scale clipping warnings.

**Test Cases**:
1. **Pointer Out-of-Range Detection**: Observe errors like "Pointer index 1373 out of range for scale 6 (size 81)" in unfixed code (will fail on unfixed code)
2. **Scale Clipping Logging**: Add logging to detect when `current_scale_label != current_scale_index + 1` (will show mismatches on unfixed code)
3. **Candidate Count Baseline**: Verify unfixed code produces 65% match rate, 890 missing candidates, 477 extra candidates (will fail on unfixed code)
4. **Trace History Analysis**: Examine trace logs showing (location, pointer, scale) tuples with mismatched pointer/scale pairs (will fail on unfixed code)

**Expected Counterexamples**:
- Pointer indices 10-100x larger than valid LUT size for their scale
- Possible causes: unclipped scale written to size_map, clipped scale used for LUT building, mismatch during backtracking

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds (out-of-range scales), the fixed function produces valid pointer indices and achieves 100% MATLAB parity.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := _matlab_global_watershed_current_strel_fixed(input)
  scale_for_writing := result["scale_label_clipped"]
  ASSERT scale_for_writing == clip(input.current_scale_label, 1, len(lumen_radius_microns))
  ASSERT all_pointers_valid_for_scale(result["pointer_indices"], scale_for_writing)
  
  // During backtracking
  reconstructed_lut := build_lut(scale_for_writing)
  ASSERT len(reconstructed_lut) == result["lut_size"]
  ASSERT all(result["pointer_indices"] <= len(reconstructed_lut))
END FOR
```

**Test Plan**: Run `capture-candidates` on fixed code and verify:
- Zero pointer out-of-range errors
- 100% candidate match rate with MATLAB
- Exactly 2533 candidates (matching MATLAB)
- Zero missing or extra candidates

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold (in-range scales), the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  // For in-range scales, clipped == unclipped
  ASSERT clip(input.current_scale_label, 1, len(lumen_radius_microns)) == input.current_scale_label
  ASSERT _matlab_global_watershed_original(input) = _matlab_global_watershed_fixed(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Run the full parity experiment suite on fixed code to verify preservation:
1. **LUT Consistency**: `prove-luts` verifies LUT building is unchanged
2. **Candidate Preservation**: `capture-candidates` verifies in-range scales produce correct candidates
3. **Edge Replay**: `replay-edges` verifies edge tracing logic is unchanged
4. **Exact Parity**: `prove-exact` verifies all normalized checkpoints match MATLAB vectors

**Test Cases**:
1. **In-Range Scale Preservation**: Verify locations with `current_scale_label` in `[1, len(lumen_radius_microns)]` produce identical pointers and size_map values
2. **Backtracking Preservation**: Verify traces from in-range scale locations are unchanged
3. **Energy Adjustment Preservation**: Verify frontier energy adjustments are unchanged
4. **Edge Metric Preservation**: Verify edge metrics calculated from energy traces are unchanged

### Unit Tests

- Test `_matlab_global_watershed_current_strel` with out-of-range scale labels (0, 12, 100) and verify clipped scale is returned
- Test pointer index validity for clipped scales (all pointers in range [1, lut_size])
- Test size_map writing with clipped scale values
- Test backtracking with clipped scales reconstructs correct LUT

### Property-Based Tests

- Generate random scale labels (including out-of-range values) and verify pointer indices are always valid for the clipped scale's LUT
- Generate random watershed states and verify backtracking never encounters out-of-range pointer errors
- Test that clipped scale in size_map always matches the scale used for pointer creation

### Integration Tests

- Run full parity experiment: `preflight-exact`, `prove-luts`, `capture-candidates`, `replay-edges`, `prove-exact`
- Verify 100% match rate with MATLAB (2533 candidates)
- Verify zero pointer out-of-range errors in logs
- Verify all diagnostic assertions pass (no LUT mismatches, no overwrites, no corruption)
