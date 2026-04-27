# v22 Blocking Bugs

[Up: Reference Docs](../README.md)

**Date**: April 27, 2026  
**Status**: BLOCKING — v22 cannot generate valid candidates

## Summary

The v22 global watershed implementation has critical bugs that prevent valid
candidate generation. A `capture-candidates` run on the native-first exact route
revealed two classes of errors that occur during candidate trace-back:

1. **Cycle detection errors** (40+ instances)
2. **Pointer index out-of-range errors** (15+ instances)

These bugs block all downstream exact-parity work until fixed.

## Error Details

### 1. Cycle Detection in Backtracking

**Error Pattern**:
```
ERROR:root:Cycle detected in global watershed backtrack at <location>. Breaking.
```

**Frequency**: 40+ instances across the canonical sample

**Description**: The backtracking logic encounters cycles in the pointer map,
causing premature termination of candidate traces. A cycle means that following
the pointer chain leads back to a previously visited location, creating an
infinite loop.

**Likely Cause**: The frontier propagation is writing pointers that create
circular references. This could happen if:
- A location is revisited during frontier expansion and overwrites its pointer
- The LIFO tie-breaking allows a location to be processed multiple times
- The energy-tolerance logic permits frontier growth that creates cycles

### 2. Pointer Index Out-of-Range

**Error Pattern**:
```
ERROR:root:Pointer index <N> out of range for scale <S> (size <M>) at <location>.
```

**Examples**:
- `Pointer index 1373 out of range for scale 6 (size 81)`
- `Pointer index 76 out of range for scale 5 (size 57)`
- `Pointer index 3099 out of range for scale 6 (size 81)`

**Frequency**: 15+ instances across multiple scales

**Description**: The pointer map contains index values that exceed the LUT size
for the given scale. For example, scale 6 has an 81-element LUT, but a pointer
value of 1373 was written.

**Likely Cause**: The pointer write logic is not respecting LUT bounds. This
could happen if:
- The linear offset calculation is incorrect
- The strel LUT indexing is off-by-one or uses the wrong scale
- The flat-first 1D architecture has an indexing bug

## Impact

- **Candidate generation**: BLOCKED — cannot produce valid candidates
- **Exact proof**: BLOCKED — no candidates to compare against MATLAB oracle
- **Downstream work**: BLOCKED — conflict painting, cleanup, network all depend
  on valid candidates

## LUT Proof Status

The `prove-luts` command passed for all 15 scales, confirming that the LUT
construction is correct. The bugs are in the runtime frontier propagation and
pointer writes, not in the LUT generation.

## Diagnostic Steps

1. **Add defensive logging**:
   - Log the pointer map state when a cycle or out-of-range error is detected
   - Capture the frontier state (energy, location, scale) at the error point
   - Record the pointer chain leading to the error

2. **Verify pointer write bounds**:
   - Check that all pointer writes use `pointer_value < lut_size[scale]`
   - Verify the linear offset calculation matches MATLAB's formula
   - Confirm the strel LUT indexing uses the correct scale

3. **Check tie-breaking and energy-tolerance logic**:
   - Verify that LIFO tie-breaking doesn't allow duplicate processing
   - Check if energy-tolerance permits frontier growth that creates cycles
   - Confirm that the ownership-reveal timing prevents self-cycles

4. **Run small-scale debug trace**:
   - Isolate the first failing candidate
   - Step through the frontier propagation for that candidate
   - Identify the exact pointer write that creates the cycle or out-of-range value

## Related Files

- `source/core/_edge_candidates/global_watershed.py` — frontier propagation
- `source/core/_edge_candidates/common.py` — LUT construction
- `dev/scripts/cli/parity_experiment.py` — proof harness
- `docs/reference/core/EXACT_PROOF_FINDINGS.md` — updated status

## Next Steps

1. Fix the cycle detection and pointer out-of-range bugs
2. Re-run `capture-candidates` to verify the fixes
3. Document the root cause and the fix in this file
4. Update `EXACT_PROOF_FINDINGS.md` with the new status
5. Resume exact-parity work once candidates are valid
