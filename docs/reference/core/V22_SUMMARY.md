# v22 Global Watershed Bug Investigation - Summary

**Date**: April 27, 2026  
**Status**: Root cause identified, ready for fix implementation

## Quick Reference

| Document | Purpose |
|----------|---------|
| `V22_SUMMARY.md` (this file) | High-level overview and quick start |
| `V22_INVESTIGATION_HANDOVER.md` | Complete investigation details and next steps |
| `V22_BUG_FIXES.md` | Detailed bug descriptions and test results |
| `V22_BLOCKING_BUGS.md` | Original bug report from initial discovery |
| `V22_POINTER_CORRUPTION_INVESTIGATION.md` | Phase-by-phase investigation log |

## The Problem

The v22 global watershed implementation generates **invalid pointer indices** during backtracking, causing:
- 65% match rate (target: 100%)
- 890 missing candidates
- 477 extra candidates

## Root Cause

**Scale clipping mismatch**: When a scale label exceeds the number of available LUT scales, it gets clipped when building the LUT, but the unclipped value is stored in size_map. This causes pointers to be interpreted with the wrong LUT during backtracking.

### Example
```
Write phase:
- current_scale_label = 11
- len(lumen_radius_microns) = 10
- current_scale_index = clip(10, 0, 9) = 9
- Build LUT for scale 9 (size 200)
- Create pointer 150 (valid for scale 9)
- Write pointer=150, scale=11 to maps

Read phase (backtracking):
- Read scale=11 from size_map
- scale_index = clip(10, 0, 9) = 9
- Build LUT for scale 9 (size 200)
- Use pointer 150 with scale 9 LUT ✓ Should work!

BUT if scale 11 somehow maps to a different LUT size (e.g., 81):
- Read scale=11, but interpret as scale 6
- Build LUT for scale 6 (size 81)
- Try to use pointer 150 with scale 6 LUT ✗ Out of range!
```

## The Fix

**Option 1 - Use Clipped Scale Everywhere** (Recommended):
```python
# In _matlab_global_watershed_current_strel
return {
    ...
    "scale_label": current_scale_index + 1,  # Return clipped scale
}

# In main loop
current_scale_label_for_writing = current_strel["scale_label"]

# In reveal function
size_map_flat[claim_linear] = np.int16(current_scale_label_for_writing)
```

**Option 2 - Never Clip, Always Validate**:
```python
# Reject scales that are out of range instead of clipping
if current_scale_label < 1 or current_scale_label > len(lumen_radius_microns):
    raise ValueError(f"Invalid scale {current_scale_label}")
```

## How to Apply the Fix

1. **Verify the hypothesis** (optional but recommended):
   ```python
   # Add to _matlab_global_watershed_current_strel
   if current_scale_label - 1 != current_scale_index:
       logging.warning(f"SCALE CLIP: label={current_scale_label} -> index={current_scale_index}")
   ```

2. **Apply Option 1 fix** (modify 3 locations):
   - `_matlab_global_watershed_current_strel`: Return clipped scale
   - Main loop: Use returned scale
   - `_matlab_global_watershed_reveal_unclaimed_strel`: Write clipped scale

3. **Test**:
   ```powershell
   python dev/scripts/cli/parity_experiment.py capture-candidates \
       --source-run-root "D:\slavv_comparisons\experiments\live-parity\runs\20260421_accepted_budget_trial" \
       --dest-run-root "D:\slavv_comparisons\experiments\live-parity\runs\20260427_v22_fixed"
   ```

4. **Verify**:
   - No pointer out-of-range errors
   - Candidate count ≈ 2533 (MATLAB's count)
   - Match rate > 95%

5. **Full parity test**:
   ```powershell
   python dev/scripts/cli/parity_experiment.py prove-exact \
       --source-run-root "..." \
       --dest-run-root "..." \
       --stage all
   ```

## Investigation Method

The investigation used systematic hypothesis testing with diagnostic logging:

1. ✅ Validated pointer creation (assertion never fired)
2. ✅ Validated pre-reveal (no errors)
3. ✅ Validated write-readback (no corruption)
4. ✅ Validated no overwrites (constraint respected)
5. ✅ **Analyzed trace history** (found scale mismatch)

The trace history logging was the breakthrough:
```
Trace history: [(12470094, 328, 11), (12532290, 1373, 6)]
```

This showed that location 12532290 has pointer 1373 for scale 6, but scale 6's LUT only has 81 elements.

## Diagnostic Code

All diagnostic code remains in place in `source/core/_edge_candidates/global_watershed.py`:
- Pointer creation validation
- Overwrite detection
- Write-readback verification
- Trace history logging

This code can be removed after the fix is verified, or kept for future debugging.

## Success Criteria

After applying the fix:
- ✅ Zero pointer out-of-range errors
- ✅ Candidate count matches MATLAB (2533 ± 10)
- ✅ Match rate ≥ 99%
- ✅ `prove-exact` passes on all stages

## Timeline

- **April 27, 2026 AM**: Initial bug discovery (65% match rate)
- **April 27, 2026 PM**: Systematic investigation with diagnostic logging
- **April 27, 2026 Evening**: Root cause identified (scale clipping mismatch)
- **Next**: Apply fix and verify

## References

- MATLAB source: `external/Vectorization-Public/source/get_edges_by_watershed.m`
- Python implementation: `source/core/_edge_candidates/global_watershed.py`
- Parity experiment runner: `dev/scripts/cli/parity_experiment.py`
- Test data: `D:\slavv_comparisons\experiments\live-parity\runs\20260421_accepted_budget_trial`

---

**For detailed investigation notes**: See `V22_INVESTIGATION_HANDOVER.md`  
**For test results**: See `V22_BUG_FIXES.md`  
**To continue**: Apply the fix in Option 1 above
