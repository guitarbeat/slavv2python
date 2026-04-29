# Investigation Summary: size_map Write Persistence Issue

## Problem Statement

Location 12532290 is written with scale=13 but reads as scale=6 during backtracking, causing pointer out-of-range errors.

## Evidence

1. **Initial Write (Correct)**:
   - Location 12532290 starts with: pointer=0, scale=6 (initialization), vertex=0
   - Gets written with: pointer=1373, scale=13, vertex=481
   - Readback immediately after write confirms: pointer=1373, scale=13, vertex=481

2. **Backtracking (Incorrect)**:
   - During trace, reads: pointer=1373, scale=6
   - Pointer 1373 is out of range for scale 6's LUT (size 81)
   - Should be in range for scale 13's LUT (size 2301)

## Root Cause Analysis

### Array Aliasing (Ruled Out)
- Verified that `size_map_flat = size_map.ravel(order="F")` creates a VIEW, not a copy
- Writes to `size_map_flat` correctly modify the original `size_map` array
- This is NOT the issue

### Write Persistence (Confirmed)
- The write DOES succeed initially (confirmed by readback)
- The value changes LATER, between the write and the backtracking
- This indicates an **overwrite bug**

### Overwrite Prevention Logic (Bug Found)
- The `is_without_vertex` check at line 218 should prevent overwrites:
  ```python
  is_without_vertex = (vertex_index_map_flat[valid_linear] == 0) & (pointer_map_flat[valid_linear] == 0)
  ```
- However, overwrite detection at line 269 was triggering, indicating locations with existing pointers were getting through
- **Critical Bug**: The overwrite detection logged an error but DID NOT prevent the write
- The writes at lines 289-293 happened regardless of whether we were overwriting existing data

## Fix Implemented

### Primary Fix: Prevent Overwrites
Modified the reveal function to actively filter out locations with existing pointers BEFORE writing:

```python
# CRITICAL FIX: Filter out locations with existing pointers to prevent overwrites
existing_pointers = pointer_map_flat[claim_linear]
no_existing_pointer_mask = existing_pointers == 0

if not np.all(no_existing_pointer_mask):
    # Log the attempted overwrite
    logging.error(f"CRITICAL: Prevented overwrite of {overwrite_count} existing pointers...")
    
    # Filter out locations with existing pointers
    claim_linear = claim_linear[no_existing_pointer_mask]
    claim_pointers = claim_pointers[no_existing_pointer_mask]
    
    # Update is_without_vertex to reflect the additional filtering
    is_without_vertex_no_overwrite = is_without_vertex.copy()
    is_without_vertex_no_overwrite[is_without_vertex] = no_existing_pointer_mask
    is_without_vertex = is_without_vertex_no_overwrite
```

### Secondary Fix: Enhanced Diagnostics
Added logging to track when location 12532290 appears in a strel:
- Logs existing values (vertex, pointer, scale) at the location
- Logs the values that would be written
- This will help identify if the `is_without_vertex` check is failing

## Expected Outcome

With this fix:
1. Locations with existing pointers will be filtered out before writing
2. No overwrites should occur
3. The scale value written during the initial write should persist
4. Backtracking should read the correct scale value
5. Pointer out-of-range errors should be eliminated

## Testing Plan

1. Run the capture-candidates integration test with the fix
2. Check logs for "CRITICAL: Prevented overwrite" messages
3. Verify that location 12532290 is NOT overwritten
4. Verify that backtracking reads scale=13 (not scale=6)
5. Verify that the match rate improves from 65% to closer to 100%

## Open Questions

1. **Why is `is_without_vertex` failing?**
   - The check should prevent locations with existing pointers from being selected
   - But the overwrite detection is triggering, indicating the check is failing
   - Possible causes:
     - Bug in the defensive pointer filtering (lines 240-262)
     - Race condition (unlikely in single-threaded code)
     - Logic error in the `is_without_vertex` calculation

2. **Is this the only overwrite bug?**
   - This fix addresses the symptom (overwrites happening)
   - But doesn't address the root cause (why `is_without_vertex` is failing)
   - There may be other locations affected by the same bug

## Next Steps

1. Run integration test with enhanced logging
2. Analyze logs to understand why `is_without_vertex` is failing
3. If overwrites are still occurring, investigate the defensive filtering logic
4. If overwrites are prevented but match rate doesn't improve, investigate other issues
