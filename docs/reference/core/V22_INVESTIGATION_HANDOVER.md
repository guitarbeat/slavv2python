# v22 Global Watershed Investigation - Handover Document

**Date**: April 27, 2026  
**Investigator**: AI Assistant  
**Status**: ACTIVE - Root cause identified, fix in progress

## Executive Summary

The v22 global watershed implementation has a critical bug causing invalid pointer indices during backtracking. This results in:
- **65% match rate** vs. target 100% exact parity with MATLAB
- **890 missing candidates** (35% of MATLAB's 2533 total)
- **477 extra candidates**

Through systematic investigation with diagnostic logging, I've identified that pointers are valid when written but become invalid when read during backtracking. The root cause appears to be a mismatch between the scale used to create pointers and the scale stored in size_map.

## Problem Statement

### Symptoms
During backtracking in `_matlab_global_watershed_trace_half`, we encounter pointer indices that are way out of range:
- Pointer 1373 for scale 6 (valid range [1, 81])
- Pointer 3099 for scale 6 (valid range [1, 81])
- Pointer 1547 for scale 1 (valid range [1, 27])

These values are 10-100x larger than valid, suggesting they were created for a different scale's LUT.

### Example Trace
```
ERROR: Pointer index 1373 out of range for scale 6 (size 81) at location 12532290.
Trace history: [(12470094, 328, 11), (12532290, 1373, 6)]
```

**Analysis**:
- Location 12532290 has pointer=1373, scale=6
- Scale 6 LUT has size 81, so valid pointers are [1, 81]
- Pointer 1373 is 17x too large
- Previous location 12470094 has pointer=328, scale=11 (likely valid for scale 11's larger LUT)

## Investigation Timeline

### Phase 1: Ruled Out Defensive Filtering
**Hypothesis**: Bounds checking was rejecting valid pointers  
**Result**: ❌ Pointers are invalid BEFORE filtering

### Phase 2: Validated Pointer Creation
**Test**: Added assertion in `_matlab_global_watershed_current_strel`  
**Result**: ✅ Assertion never fires - pointers are valid when created

### Phase 3: Validated Pre-Reveal
**Test**: Added validation before calling reveal function  
**Result**: ✅ No errors - pointers are valid before reveal

### Phase 4: Write-Readback Verification
**Test**: Immediate readback after writing to pointer_map_flat  
**Result**: ✅ No corruption - pointers read back correctly

### Phase 5: Overwrite Detection
**Test**: Check if locations are being claimed multiple times  
**Result**: ✅ No overwrites detected - `is_without_vertex` constraint is respected

### Phase 6: Trace History Analysis
**Test**: Log the backtracking path showing (location, pointer, scale) tuples  
**Result**: ✅ **SMOKING GUN FOUND** - Pointers don't match their scale's LUT size

## Root Cause Hypothesis

The most likely cause is a **scale mismatch** during pointer creation and storage:

1. **Scenario A - Scale Clipping Bug**:
   - `current_scale_label` might exceed `len(lumen_radius_microns)`
   - Scale gets clipped when building LUT: `clip(scale_label - 1, 0, max_scale)`
   - Pointers are created for the clipped scale's LUT
   - But `current_scale_label` (unclipped) is written to size_map
   - During backtracking, we read the unclipped scale and build wrong LUT

2. **Scenario B - Multi-Scale Strel Bug**:
   - A location might be reached from multiple scales
   - First write: pointer for scale A, size_map = A
   - Second write: only size_map gets updated to scale B, but pointer remains from scale A
   - But this contradicts our overwrite detection...

3. **Scenario C - Initial size_map Contamination**:
   - The input `scale_indices` might have values that don't match the LUT scales
   - When we read `current_scale_label` from size_map, it might not correspond to any valid LUT
   - We build a LUT for a different scale than what's stored

## Diagnostic Code Added

All diagnostic code is in `source/core/_edge_candidates/global_watershed.py`:

1. **Line ~140**: Assertion in `_matlab_global_watershed_current_strel` to verify pointer indices
2. **Line ~220**: Array length mismatch detection in reveal function
3. **Line ~230**: Overwrite detection (checks existing_pointers != 0)
4. **Line ~240**: Write-readback verification
5. **Line ~250**: Sample write logging (0.01% of writes)
6. **Line ~560**: Pre-reveal validation in main loop
7. **Line ~420**: Trace history logging in backtracking function

## Files Modified

### Source Code
- `source/core/_edge_candidates/global_watershed.py` - Added comprehensive diagnostics

### Documentation
- `docs/reference/core/V22_BUG_FIXES.md` - Bug fix tracking
- `docs/reference/core/V22_BLOCKING_BUGS.md` - Original bug report
- `docs/reference/core/V22_POINTER_CORRUPTION_INVESTIGATION.md` - Detailed investigation log
- `docs/reference/core/V22_INVESTIGATION_HANDOVER.md` - This document

## Next Steps for Resolution

### Immediate Actions

1. **Verify Scale Clipping Hypothesis**:
   ```python
   # Add logging in _matlab_global_watershed_current_strel
   if current_scale_label - 1 != current_scale_index:
       logging.warning(f"Scale clipping: label={current_scale_label}, index={current_scale_index}")
   ```

2. **Check Input scale_indices**:
   - Log the range of values in the input `scale_indices` array
   - Verify they're all in valid range [0, len(lumen_radius_microns)-1]

3. **Add Scale Consistency Check**:
   - Before writing pointers, verify that `current_scale_label - 1 == current_scale_index`
   - If not, use the clipped index for both LUT building AND size_map writing

### Proposed Fix

If Scenario A is correct, the fix is:

```python
# In _matlab_global_watershed_current_strel, return the CLIPPED scale
return {
    ...
    "lut_size": len(offsets),
    "scale_label": current_scale_index + 1,  # Return clipped scale, not input scale
}

# In main loop, use the returned scale
current_scale_label_for_writing = current_strel["scale_label"]

# In reveal function, write the clipped scale
size_map_flat[claim_linear] = np.int16(current_scale_label_for_writing)
```

### Testing Plan

1. Run `capture-candidates` with scale clipping logging
2. Verify if scale clipping is occurring
3. Apply the fix
4. Run `capture-candidates` again and verify:
   - No more pointer out-of-range errors
   - Candidate count closer to MATLAB's 2533
   - Match rate improves from 65% toward 100%
5. Run full parity experiment: `preflight-exact`, `prove-luts`, `capture-candidates`, `replay-edges`, `prove-exact`

## How to Continue Investigation

### Run Diagnostic Test
```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD"
python dev/scripts/cli/parity_experiment.py capture-candidates `
    --source-run-root "D:\slavv_comparisons\experiments\live-parity\runs\20260421_accepted_budget_trial" `
    --dest-run-root "D:\slavv_comparisons\experiments\live-parity\runs\20260427_v22_scale_check"
```

### Check for Scale Clipping
```powershell
# Look for scale clipping warnings in output
Get-Content "D:\slavv_comparisons\experiments\live-parity\runs\20260427_v22_scale_check\*.log" | Select-String "Scale clipping"
```

### Analyze Results
```powershell
# Check candidate coverage
python dev/scripts/cli/parity_experiment.py summarize `
    --run-root "D:\slavv_comparisons\experiments\live-parity\runs\20260427_v22_scale_check"
```

## Key Code Locations

### Pointer Creation
**File**: `source/core/_edge_candidates/global_watershed.py`  
**Function**: `_matlab_global_watershed_current_strel` (lines ~100-180)
```python
current_scale_index = int(np.clip(int(current_scale_label) - 1, 0, len(lumen_radius_microns) - 1))
lut = _build_matlab_global_watershed_lut(current_scale_index, ...)
pointer_indices = np.arange(1, len(offsets) + 1, dtype=np.uint64)[valid_mask]
```

### Pointer Writing
**File**: `source/core/_edge_candidates/global_watershed.py`  
**Function**: `_matlab_global_watershed_reveal_unclaimed_strel` (lines ~190-270)
```python
pointer_map_flat[claim_linear] = claim_pointers
size_map_flat[claim_linear] = np.int16(current_scale_label)  # ← Potential bug here
```

### Pointer Reading
**File**: `source/core/_edge_candidates/global_watershed.py`  
**Function**: `_matlab_global_watershed_trace_half` (lines ~400-480)
```python
pointer_value = int(pointer_map_flat[tracing_linear])
tracing_scale_label = int(size_map_flat[tracing_linear])
tracing_scale_index = int(np.clip(tracing_scale_label - 1, 0, len(lumen_radius_microns) - 1))
lut = _build_matlab_global_watershed_lut(tracing_scale_index, ...)
```

## MATLAB Reference

**File**: `external/Vectorization-Public/source/get_edges_by_watershed.m`  
**Key sections**:
- Line ~70: `strel_pointers_LUT_range = cellfun(@(x) transpose(1:numel(x)), ...)`
- Line ~428: `pointer_map(current_strel_unclaimed) = current_strel_pointers_LUT_range(...)`

The MATLAB code uses cell arrays for LUTs, one per scale. Pointers are always 1-based indices into the appropriate scale's LUT.

## Contact & Handover

**Current State**: Diagnostic code is in place, trace history is being logged, root cause hypothesis is scale clipping.

**To Resume**: 
1. Check the process output for scale clipping warnings
2. If confirmed, apply the proposed fix
3. Run full parity test suite

**Questions**: Check the investigation log files in `docs/reference/core/V22_*.md`

---

**Last Updated**: April 27, 2026  
**Next Review**: After scale clipping verification
