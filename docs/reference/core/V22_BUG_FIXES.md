# v22 Bug Fixes

[Up: Reference Docs](../README.md)

**Date**: April 27, 2026  
**Status**: IN PROGRESS — Fixes applied, testing in progress

## Bugs Fixed

### 1. Improved Backtracking Bounds Checking

**File**: `source/core/_edge_candidates/global_watershed.py`  
**Function**: `_matlab_global_watershed_trace_half`

**Changes**:
- Added explicit bounds check after computing the next tracing location
- Added more detailed error messages showing valid ranges and actual values
- Added check for out-of-bounds locations after applying the offset

**Code**:
```python
# Bounds check the new location
if tracing_linear < 0 or tracing_linear >= img_size:
    logging.error(
        f"Backtrack resulted in out-of-bounds location {tracing_linear} "
        f"(image size {img_size}) from pointer {pointer_value} with offset {offset}"
    )
    break
```

### 2. Added Pointer Validation in Reveal Function

**File**: `source/core/_edge_candidates/global_watershed.py`  
**Function**: `_matlab_global_watershed_reveal_unclaimed_strel`

**Changes**:
- Added `lut_size` parameter to validate pointer indices before writing
- Added defensive check to filter out invalid pointers before writing to pointer_map
- Added error logging when invalid pointers are detected

**Code**:
```python
# Validate pointer indices before writing
if np.any(claim_pointers < 1) or np.any(claim_pointers > lut_size):
    logging.error(
        f"Attempting to write invalid pointers: {bad_pointers[:10]} "
        f"(LUT size={lut_size}, scale={current_scale_label})"
    )
    # Filter out invalid pointers
    valid_pointer_mask = (claim_pointers >= 1) & (claim_pointers <= lut_size)
    ...
```

### 3. Added LUT Consistency Checks

**File**: `source/core/_edge_candidates/global_watershed.py`  
**Function**: `_matlab_global_watershed_current_strel`

**Changes**:
- Added assertion to verify `local_subscripts` and `linear_offsets` have the same length
- Added assertion to verify all pointer indices are in valid range [1, lut_size]
- Added `lut_size` to the return dictionary for downstream validation

**Code**:
```python
# Verify LUT consistency
assert len(offsets) == len(linear_offsets_full), (
    f"LUT inconsistency: local_subscripts has {len(offsets)} elements "
    f"but linear_offsets has {len(linear_offsets_full)} elements"
)

# Verify all pointer indices are valid
assert np.all(pointer_indices >= 1) and np.all(pointer_indices <= len(offsets)), (
    f"Invalid pointer indices: min={np.min(pointer_indices)}, "
    f"max={np.max(pointer_indices)}, LUT size={len(offsets)}"
)
```

## Root Cause Analysis

The bugs were caused by:

1. **Insufficient bounds checking**: The backtracking logic didn't verify that the computed next location was within the image bounds after applying the pointer offset.

2. **No validation of pointer writes**: The reveal function was writing pointer indices without verifying they were valid for the LUT size of the current scale.

3. **Potential LUT inconsistencies**: There was no verification that the LUT components (local_subscripts, linear_offsets) were consistent with each other.

## Testing Status

**Test Run**: `capture-candidates` on native-first exact route  
**Start Time**: April 27, 2026 1:30 PM  
**Status**: IN PROGRESS (command still running after 5 minutes)

The fixes prevent the immediate crashes, but the algorithm is taking significantly longer than expected. This suggests either:
- The defensive checks are catching and filtering many invalid pointers (good - prevents crashes)
- The algorithm has performance issues that need optimization
- There may be additional logic bugs causing excessive iterations

## Next Steps

1. Wait for the current `capture-candidates` run to complete or timeout
2. Examine the output to see if valid candidates were generated
3. Check logs for how many invalid pointers were filtered
4. If still failing, add more detailed logging to trace the frontier propagation
5. Consider adding iteration limits or progress monitoring to prevent infinite loops

## Related Files

- `source/core/_edge_candidates/global_watershed.py` — main implementation
- `source/core/_edge_candidates/common.py` — LUT construction
- `docs/reference/core/V22_BLOCKING_BUGS.md` — original bug report
- `docs/reference/core/EXACT_PROOF_FINDINGS.md` — proof status
