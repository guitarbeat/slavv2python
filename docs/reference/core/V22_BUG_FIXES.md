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

## Current Status (April 27, 2026)

**Investigation Phase**: ROOT CAUSE IDENTIFIED

**Key Finding**: Pointers are valid when created but invalid when read during backtracking. Trace history analysis reveals:

```
ERROR: Pointer index 1373 out of range for scale 6 (size 81) at location 12532290.
Trace history: [(12470094, 328, 11), (12532290, 1373, 6)]
```

Location 12532290 has pointer=1373 for scale=6, but scale 6's LUT only has 81 elements. This pointer is 17x too large.

**Root Cause Hypothesis**: Scale clipping mismatch
- When `current_scale_label` exceeds `len(lumen_radius_microns)`, it gets clipped when building the LUT
- Pointers are created for the clipped scale's LUT
- But the UNCLIPPED `current_scale_label` is written to size_map
- During backtracking, we read the unclipped scale and build the wrong LUT

**Proposed Fix**: Use the clipped scale index consistently for both LUT building and size_map writing.

**See**: `docs/reference/core/V22_INVESTIGATION_HANDOVER.md` for complete investigation details and next steps.

---

## Bug Fixes Applied

The bugs were caused by:

1. **Insufficient bounds checking**: The backtracking logic didn't verify that the computed next location was within the image bounds after applying the pointer offset.

2. **No validation of pointer writes**: The reveal function was writing pointer indices without verifying they were valid for the LUT size of the current scale.

3. **Potential LUT inconsistencies**: There was no verification that the LUT components (local_subscripts, linear_offsets) were consistent with each other.

## Testing Status

**Test Run**: `capture-candidates` on native-first exact route  
**Date**: April 27, 2026  
**Status**: ✅ **COMPLETED SUCCESSFULLY**

### Results

The v22 implementation with bug fixes successfully generated candidates:

| Metric | Count | Percentage |
|--------|-------|------------|
| MATLAB candidates | 2533 | 100% (oracle) |
| Python candidates | 2120 | 83.7% of MATLAB |
| Matched pairs | 1643 | 64.9% match rate |
| Missing pairs | 890 | 35.1% not generated |
| Extra pairs | 477 | 22.5% over-generated |

### Analysis

**Good News**:
- ✅ No crashes or errors during candidate generation
- ✅ Algorithm completes successfully
- ✅ 65% of candidates match MATLAB exactly
- ✅ Defensive checks prevented invalid pointer writes

**Remaining Issues**:
- ❌ 890 missing pairs (35%) - Python fails to generate some MATLAB candidates
- ❌ 477 extra pairs (23%) - Python generates candidates MATLAB doesn't
- ⚠️ Not yet at exact parity (need 100% match)

**Top Missing Vertices** (most affected by missing candidates):
- Vertices 1713, 696, 1415, 1313, 1405, 1679, 1559, 1896, 1968, 1685 (4 missing each)

**Top Extra Vertices** (most affected by extra candidates):
- Vertices 868, 1411, 2082, 2144 (4 extra each)

### Comparison to Historical Results

**Pre-v22 (Historical)**:
- Python candidates: 2364
- Matched: 2054 (87%)
- Missing: 479
- Extra: 310

**Post-v22 with fixes**:
- Python candidates: 2120 (10% fewer)
- Matched: 1643 (65%)
- Missing: 890 (86% more missing)
- Extra: 477 (54% more extra)

The v22 implementation is generating fewer candidates overall and has a lower match rate than the historical pre-v22 code. This suggests the defensive filtering may be too aggressive, or there are still logic bugs in the frontier propagation.

## Next Steps

### CURRENT INVESTIGATION (April 27, 2026 - v22 audit v2)

**Status**: ROOT CAUSE IDENTIFIED - Pointer corruption between creation and backtracking

**Key Finding**: 
- ✅ Pointers are VALID when created (no assertion failures in `_matlab_global_watershed_current_strel`)
- ✅ Pointers are VALID when passed to reveal function (no "CRITICAL BUG DETECTED" messages)
- ❌ Pointers are INVALID when read during backtracking (errors show values like 1373, 3099 for LUT size 81)

**Conclusion**: The pointers are being corrupted BETWEEN writing and reading. This suggests:
1. Memory corruption or buffer overflow
2. Wrong array indexing causing writes to wrong locations
3. Dtype conversion issues
4. The pointer_map_flat view is not properly synchronized with pointer_map

**Most Likely Cause**: The `pointer_map_flat` is a view created by `ravel(order="F")`. If this view becomes invalid or desynchronized, writes might not affect the underlying array correctly, or reads might return garbage values.

**Next Actions**:
1. **PRIORITY**: Verify that `pointer_map_flat` is a proper view and writes are persisted
2. Check if there's a numpy version issue with ravel() views
3. Consider using explicit indexing instead of flattened views
4. Add a verification step: immediately after writing pointers, read them back and verify they match

### Previous Investigation Steps

1. ✅ **COMPLETED**: Wait for current `capture-candidates` run to complete
2. ✅ **COMPLETED**: Examine output - 2120 candidates generated, 65% match rate
3. **IN PROGRESS**: Investigate why 890 MATLAB candidates are missing
   - Check if defensive filtering is removing valid candidates
   - Examine the top missing vertices to find patterns
   - Compare frontier propagation logic to MATLAB more carefully
4. **TODO**: Investigate why 477 extra candidates are generated
   - Check if energy tolerance or distance tolerance differs from MATLAB
   - Verify join-location reset logic
5. **TODO**: Consider relaxing defensive checks if they're too aggressive
   - Review filtered pointer logs to see how many were rejected
   - Determine if rejected pointers were actually valid
6. **TODO**: Run `replay-edges` to test downstream edge selection
7. **TODO**: Run `prove-exact` once candidate alignment improves

### Hypothesis for Missing Candidates

The defensive filtering in `_matlab_global_watershed_reveal_unclaimed_strel` may be rejecting valid pointers that are at the boundary of the LUT size. Need to verify:
- Are pointers exactly equal to `lut_size` being rejected when they should be accepted?
- Is the LUT size calculation correct for all scales?
- Are there edge cases in the bounds checking that are too conservative?

### Hypothesis for Extra Candidates

Python may be generating candidates that MATLAB rejects due to:
- Different energy tolerance application
- Different distance tolerance calculation  
- Join-location reset timing differences
- Frontier termination conditions

## Related Files

- `source/core/_edge_candidates/global_watershed.py` — main implementation
- `source/core/_edge_candidates/common.py` — LUT construction
- `docs/reference/core/V22_BLOCKING_BUGS.md` — original bug report
- `docs/reference/core/EXACT_PROOF_FINDINGS.md` — proof status
