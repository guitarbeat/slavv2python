# Task 3.5 Investigation Findings

## Problem Summary

The fix for the v22 pointer corruption bug has been implemented correctly:
- `_matlab_global_watershed_current_strel` returns `scale_label_clipped` (line 187-188)
- Main loop extracts and uses the clipped scale for writing (lines 738-739, 793)
- Unit tests pass for both bug condition and preservation properties

However, the integration test (capture-candidates) shows NO improvement:
- Still 65% match rate (1643/2533 candidates)
- Still 890 missing candidates, 477 extra candidates
- Pointer out-of-range errors still occurring

## Diagnostic Findings

### Key Observation from Logs

Location 12532290 demonstrates the problem:

1. **Initial Write (Correct)**:
   ```
   ABOUT TO WRITE TO LOCATION 12532290: will_write_pointer=1373, will_write_scale=13, lut_size=2301, will_write_vertex=481
   READBACK BEFORE WRITE: current_pointer=0, current_scale=6, current_vertex=0
   READBACK AFTER WRITE: pointer_map[12532290]=1373, size_map[12532290]=13, vertex_index_map[12532290]=481
   ```
   - Location starts with scale=6 (from initialization)
   - Gets written with pointer=1373, scale=13, vertex=481
   - Readback confirms write succeeded

2. **Backtracking (Incorrect)**:
   ```
   ERROR:root:Pointer index 1373 out of range for scale 6 (size 81) at location 12532290. 
   Valid range: [1, 81]. Trace history: [(12470094, 328, 11), (12532290, 1373, 6)]
   ```
   - Backtracking reads pointer=1373, scale=6 from location 12532290
   - But we wrote scale=13!
   - This causes pointer 1373 to be out of range for scale 6's LUT (size 81)

### Root Cause Hypothesis

The scale value in `size_map` is being **overwritten** after the initial correct write. Possible causes:

1. **Overwrite Bug**: The `is_without_vertex` check (line 218) should prevent overwrites, but the overwrite detection (line 269) is triggering, suggesting locations with `vertex_index==0` still have non-zero pointers. This indicates `vertex_index_map` and `pointer_map` are out of sync.

2. **Multiple Writes**: Location 12532290 appears as a neighbor multiple times:
   - First from vertex 481 with scale 13 (written successfully)
   - Later from vertex 525 with scale 8 (should be blocked by `is_without_vertex`)
   
   If the second write is NOT being blocked, it would overwrite scale=13 with scale=8 (or the initialization value of 6).

3. **Defensive Filtering Bug**: Lines 256-262 filter out invalid pointers but might be incorrectly modifying the `is_without_vertex` mask, causing valid writes to be skipped or invalid writes to proceed.

## Investigation Steps Needed

1. **Check for Overwrites**: Run test with enhanced logging to see if location 12532290 is being overwritten after the initial write with scale=13.

2. **Verify is_without_vertex Logic**: Confirm that the `is_without_vertex` check is correctly preventing overwrites. The overwrite detection should NEVER trigger if this logic is correct.

3. **Check Defensive Filtering**: Verify that the defensive pointer filtering (lines 256-262) isn't corrupting the `is_without_vertex` mask or allowing invalid writes.

4. **Memory Corruption**: If no overwrites are detected, investigate potential memory corruption or array aliasing issues.

## Code Issues Found

1. **Duplicate Write Code**: Found and removed duplicate write statements that were causing confusion in diagnostic output.

2. **Overwrite Detection Doesn't Prevent Writes**: The overwrite detection at line 269 logs an error but doesn't prevent the overwrite from happening. This should either:
   - Filter out locations with existing pointers before writing
   - OR raise an exception to fail fast

3. **Diagnostic Code Bug**: Fixed UnboundLocalError where `current_strel_linear` was referenced before extraction from dictionary.

## Next Steps

1. Re-run capture-candidates test with enhanced diagnostic logging
2. Analyze logs to determine if location 12532290 is being overwritten
3. If overwrites are occurring, identify why `is_without_vertex` isn't preventing them
4. If no overwrites, investigate memory corruption or array indexing bugs
