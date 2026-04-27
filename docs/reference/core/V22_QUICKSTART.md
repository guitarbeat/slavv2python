# v22 Bug Fix - Quick Start Guide

**Last Updated**: April 27, 2026  
**Time to Fix**: ~30 minutes  
**Difficulty**: Medium

## TL;DR

The v22 global watershed has a scale clipping bug. Pointers are created for one scale but stored with a different scale label, causing out-of-range errors during backtracking.

**Fix**: Make scale clipping consistent between pointer creation and storage.

## The 3-Minute Version

### What's Wrong
```python
# Current (BROKEN):
current_scale_index = clip(current_scale_label - 1, 0, max)  # Clipped
lut = build_lut(current_scale_index)  # Uses clipped scale
pointers = create_pointers(lut)  # Pointers for clipped scale
size_map[location] = current_scale_label  # Stores UNCLIPPED scale ← BUG!

# Later during backtracking:
scale = size_map[location]  # Reads UNCLIPPED scale
lut = build_lut(clip(scale - 1, 0, max))  # Might build different LUT
use_pointer(pointer, lut)  # Pointer doesn't match LUT → ERROR!
```

### The Fix
```python
# Option 1: Store clipped scale (RECOMMENDED)
current_scale_index = clip(current_scale_label - 1, 0, max)
lut = build_lut(current_scale_index)
pointers = create_pointers(lut)
size_map[location] = current_scale_index + 1  # Store clipped scale ← FIX!

# Option 2: Never clip, validate instead
if current_scale_label < 1 or current_scale_label > len(lumen_radius_microns):
    raise ValueError(f"Invalid scale {current_scale_label}")
```

## Step-by-Step Fix (Option 1)

### Step 1: Modify `_matlab_global_watershed_current_strel`

**File**: `source/core/_edge_candidates/global_watershed.py`  
**Line**: ~170

**Change**:
```python
# OLD:
return {
    "current_coord": current_coord.astype(np.int32, copy=False),
    "coords": valid_coords,
    "offsets": valid_offsets,
    "linear_indices": valid_linear,
    "pointer_indices": pointer_indices,
    "r_over_R": np.asarray(lut["r_over_R"], dtype=np.float32)[valid_mask],
    "distance_microns": np.asarray(lut["distance_lut"], dtype=np.float32)[valid_mask],
    "unit_vectors": np.asarray(lut["unit_vectors"], dtype=np.float32)[valid_mask],
    "lut_size": len(offsets),
}

# NEW:
return {
    "current_coord": current_coord.astype(np.int32, copy=False),
    "coords": valid_coords,
    "offsets": valid_offsets,
    "linear_indices": valid_linear,
    "pointer_indices": pointer_indices,
    "r_over_R": np.asarray(lut["r_over_R"], dtype=np.float32)[valid_mask],
    "distance_microns": np.asarray(lut["distance_lut"], dtype=np.float32)[valid_mask],
    "unit_vectors": np.asarray(lut["unit_vectors"], dtype=np.float32)[valid_mask],
    "lut_size": len(offsets),
    "scale_label_clipped": current_scale_index + 1,  # ADD THIS LINE
}
```

### Step 2: Use clipped scale in main loop

**File**: `source/core/_edge_candidates/global_watershed.py`  
**Line**: ~550

**Change**:
```python
# OLD:
current_strel = _matlab_global_watershed_current_strel(
    current_linear,
    current_scale_label=current_scale_label,
    shape=shape,
    lumen_radius_microns=lumen_radius_microns,
    microns_per_voxel=microns_per_voxel,
    step_size_per_origin_radius=step_size_per_origin_radius,
)

# NEW:
current_strel = _matlab_global_watershed_current_strel(
    current_linear,
    current_scale_label=current_scale_label,
    shape=shape,
    lumen_radius_microns=lumen_radius_microns,
    microns_per_voxel=microns_per_voxel,
    step_size_per_origin_radius=step_size_per_origin_radius,
)
current_scale_label_for_writing = current_strel["scale_label_clipped"]  # ADD THIS LINE
```

### Step 3: Pass clipped scale to reveal function

**File**: `source/core/_edge_candidates/global_watershed.py`  
**Line**: ~580

**Change**:
```python
# OLD:
_matlab_global_watershed_reveal_unclaimed_strel(
    current_vertex_index=current_vertex_index,
    current_scale_label=current_scale_label,  # ← OLD
    current_d_over_r=current_d_over_r,
    ...
)

# NEW:
_matlab_global_watershed_reveal_unclaimed_strel(
    current_vertex_index=current_vertex_index,
    current_scale_label=current_scale_label_for_writing,  # ← NEW
    current_d_over_r=current_d_over_r,
    ...
)
```

### Step 4: Test

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Compile
python -m compileall source/core/_edge_candidates/global_watershed.py

# Run test
$env:PYTHONPATH = "$PWD"
python dev/scripts/cli/parity_experiment.py capture-candidates `
    --source-run-root "D:\slavv_comparisons\experiments\live-parity\runs\20260421_accepted_budget_trial" `
    --dest-run-root "D:\slavv_comparisons\experiments\live-parity\runs\20260427_v22_FIXED"
```

### Step 5: Verify

**Expected results**:
- ✅ No "Pointer index X out of range" errors
- ✅ Candidate count ≈ 2533 (MATLAB's count)
- ✅ Match rate > 95%

**Check results**:
```powershell
# Look for errors
Get-ChildItem "D:\slavv_comparisons\experiments\live-parity\runs\20260427_v22_FIXED" -Recurse | Select-String "ERROR"

# Check summary
python dev/scripts/cli/parity_experiment.py summarize `
    --run-root "D:\slavv_comparisons\experiments\live-parity\runs\20260427_v22_FIXED"
```

## If It Works

1. **Remove diagnostic code** (optional):
   - Search for "DIAGNOSTIC" comments in `global_watershed.py`
   - Remove logging statements
   - Keep the fix!

2. **Run full parity test**:
   ```powershell
   python dev/scripts/cli/parity_experiment.py prove-exact `
       --source-run-root "D:\slavv_comparisons\experiments\live-parity\runs\20260421_accepted_budget_trial" `
       --dest-run-root "D:\slavv_comparisons\experiments\live-parity\runs\20260427_v22_FINAL" `
       --stage all
   ```

3. **Update documentation**:
   - Mark `V22_SUMMARY.md` as RESOLVED
   - Update `EXACT_PROOF_FINDINGS.md` with new results
   - Update `MATLAB_METHOD_IMPLEMENTATION_PLAN.md` if needed

## If It Doesn't Work

1. **Check if scale clipping is actually happening**:
   ```python
   # Add to _matlab_global_watershed_current_strel (line ~110)
   if current_scale_label - 1 != current_scale_index:
       import logging
       logging.warning(f"SCALE CLIP: {current_scale_label} -> {current_scale_index}")
   ```

2. **Check the trace history**:
   - Look for "Trace history" in error messages
   - Verify the scale/pointer/LUT size relationships

3. **Read the full investigation**:
   - See `V22_INVESTIGATION_HANDOVER.md` for complete details
   - Check if there's a different root cause

## Common Issues

### "No such file or directory"
- Make sure you're in the repo root: `cd C:\Users\alw4834\Documents\slavv2python`
- Activate venv: `.\.venv\Scripts\Activate.ps1`

### "Module not found"
- Set PYTHONPATH: `$env:PYTHONPATH = "$PWD"`

### "Still getting pointer errors"
- Check if you modified all 3 locations
- Verify the clipped scale is being used consistently
- Check the error messages for the scale values

### "Different error now"
- This might be progress! The fix might have exposed a different bug
- Document the new error
- Check if it's a downstream issue (edges, network, etc.)

## Success Criteria

- ✅ Zero pointer out-of-range errors
- ✅ Candidate count: 2533 ± 10
- ✅ Match rate: ≥ 99%
- ✅ All parity stages pass

## Time Estimate

- Reading this guide: 5 min
- Making the changes: 10 min
- Testing: 10-15 min
- Verification: 5 min
- **Total**: ~30 minutes

## Need Help?

1. Read `V22_SUMMARY.md` for overview
2. Read `V22_INVESTIGATION_HANDOVER.md` for details
3. Check error messages against the investigation logs
4. Look for similar patterns in the MATLAB source

---

**Good luck!** This fix should resolve the 65% match rate issue and get you to exact parity.
