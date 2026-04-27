# v22 Pointer Corruption Investigation

**Date**: April 27, 2026  
**Status**: ACTIVE INVESTIGATION

## Problem Statement

The v22 global watershed implementation generates invalid pointer indices during backtracking, causing:
- 890 missing candidates (35% of MATLAB's 2533)
- 477 extra candidates  
- Overall 65% match rate vs. target 100% exact parity

## Investigation Timeline

### Phase 1: Initial Hypothesis - Defensive Filtering
**Hypothesis**: Defensive bounds checking was rejecting valid pointers  
**Result**: ❌ REJECTED - Pointers are invalid BEFORE filtering, not because of filtering

### Phase 2: Pointer Generation Validation
**Hypothesis**: Pointers are created with invalid values  
**Test**: Added assertion in `_matlab_global_watershed_current_strel` to verify pointer indices are in [1, lut_size]  
**Result**: ✅ ASSERTION NEVER FIRES - Pointers are valid when created

### Phase 3: Pre-Reveal Validation
**Hypothesis**: Pointers become invalid between creation and reveal function  
**Test**: Added validation immediately before calling `_matlab_global_watershed_reveal_unclaimed_strel`  
**Result**: ✅ NO ERRORS - Pointers are still valid before reveal

### Phase 4: Write-Readback Verification
**Hypothesis**: Pointers are corrupted during the write operation  
**Test**: Added immediate readback verification after writing to pointer_map_flat  
**Result**: ✅ NO CORRUPTION DETECTED - Pointers read back correctly immediately after writing

### Phase 5: Current Focus - Delayed Corruption
**Hypothesis**: Pointers are valid when written but become invalid later, possibly due to:
1. Size_map being overwritten with wrong scale
2. Pointer_map being overwritten by subsequent operations
3. Memory corruption or view desynchronization

**Evidence**:
- Pointers are valid at creation ✓
- Pointers are valid before reveal ✓  
- Pointers write and readback correctly ✓
- Pointers are INVALID during backtracking ✗

**Conclusion**: The corruption happens BETWEEN writing and backtracking, not during the write itself.

## Technical Details

### Pointer Semantics
- Pointers are 1-based indices into the FULL LUT for a given scale
- Valid range: [1, lut_size] where lut_size depends on the scale
- Example: Scale 6 has LUT size 81, so valid pointers are [1, 81]

### Observed Invalid Pointers
- Pointer 1373 for scale 6 (LUT size 81)
- Pointer 3099 for scale 6 (LUT size 81)
- Pointer 1547 for scale 1 (LUT size 27)
- Pointer 459 for scale 6 (LUT size 81)

These values are 10-100x larger than the valid range, suggesting they might be:
- Linear offsets (which can be large) instead of pointer indices
- Corrupted memory values
- Values from a different data structure

### Key Code Locations

**Pointer Creation**: `_matlab_global_watershed_current_strel`
```python
pointer_indices = np.arange(1, len(offsets) + 1, dtype=np.uint64)[valid_mask]
```

**Pointer Writing**: `_matlab_global_watershed_reveal_unclaimed_strel`
```python
pointer_map_flat[claim_linear] = claim_pointers
size_map_flat[claim_linear] = np.int16(current_scale_label)
```

**Pointer Reading**: `_matlab_global_watershed_trace_half`
```python
pointer_value = int(pointer_map_flat[tracing_linear])
tracing_scale_label = int(size_map_flat[tracing_linear])
```

## Next Steps

1. **Check for overwrites**: Verify that locations are not being claimed multiple times with different scales
2. **Trace specific location**: Pick one failing location and trace its full history (when written, what values, when read)
3. **Compare with MATLAB**: Verify the MATLAB code doesn't have similar issues or special handling
4. **Memory layout**: Verify that pointer_map_flat is a proper view and not getting desynchronized

## Diagnostic Code Added

- ✅ Assertion in pointer creation
- ✅ Validation before reveal
- ✅ Array length mismatch detection
- ✅ Overwrite detection (checks if existing_pointers != 0)
- ✅ Write-readback verification
- ✅ Enhanced error messages with dtype and value information

## Related Files

- `source/core/_edge_candidates/global_watershed.py` - Main implementation
- `docs/reference/core/V22_BUG_FIXES.md` - Bug fix documentation
- `docs/reference/core/V22_BLOCKING_BUGS.md` - Original bug report
