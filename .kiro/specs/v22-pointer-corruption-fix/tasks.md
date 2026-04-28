# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Out-of-Range Scale Pointer Validity
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate pointer corruption for out-of-range scales
  - **Scoped PBT Approach**: Scope the property to concrete failing cases where `current_scale_label` is outside `[1, len(lumen_radius_microns)]`
  - Test that when `current_scale_label` exceeds valid range, the system clips the scale and uses the clipped value consistently for both pointer creation and size_map storage
  - Generate test cases with `current_scale_label` values like 0, 12, 100 (outside typical range of 1-11)
  - Assert that `scale_label_clipped` returned from `_matlab_global_watershed_current_strel` equals `clip(current_scale_label - 1, 0, len(lumen_radius_microns) - 1) + 1`
  - Assert that all pointer indices are within valid range `[1, lut_size]` for the clipped scale's LUT
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists)
  - Document counterexamples found (e.g., "scale 12 clipped to 10, but unclipped scale 12 written to size_map, causing LUT mismatch during backtracking")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - In-Range Scale Behavior Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for in-range scale inputs (where `current_scale_label` is in `[1, len(lumen_radius_microns)]`)
  - For in-range scales, verify that clipped scale equals unclipped scale (no clipping occurs)
  - Write property-based test: for all `current_scale_label` in `[1, len(lumen_radius_microns)]`, the pointer creation, size_map writing, and backtracking behavior is unchanged
  - Generate test cases with valid scale labels (1 through 11 for typical configurations)
  - Assert that pointer indices remain in valid range for their scale's LUT
  - Assert that backtracking reconstructs the correct LUT and successfully traces without errors
  - Assert that LUT building produces identical results for each scale index
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [ ] 3. Fix for scale clipping inconsistency causing pointer corruption

  - [x] 3.1 Implement the fix in `_matlab_global_watershed_current_strel`
    - Modify `_matlab_global_watershed_current_strel` function (lines ~100-180 in `source/core/_edge_candidates/global_watershed.py`)
    - Add `"scale_label_clipped": current_scale_index + 1` to the return dictionary to provide the clipped scale (converting back to 1-based label)
    - This ensures the clipped scale is available for consistent use in pointer creation and size_map storage
    - _Bug_Condition: isBugCondition(input) where `(input.current_scale_label < 1 OR input.current_scale_label > len(input.lumen_radius_microns))`_
    - _Expected_Behavior: For out-of-range scales, return clipped scale and use it consistently for both LUT building and size_map writing_
    - _Preservation: In-range scales continue to work exactly as before (clipped == unclipped)_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.2 Update main loop to use clipped scale for size_map writing
    - In `_generate_edge_candidates_matlab_global_watershed` (line ~560), extract the clipped scale from the strel result
    - Add: `current_scale_label_for_writing = current_strel.get("scale_label_clipped", current_scale_label)`
    - Update the call to `_matlab_global_watershed_reveal_unclaimed_strel` to use `current_scale_label_for_writing` instead of `current_scale_label`
    - This ensures the clipped scale is written to size_map, matching the scale used for pointer creation
    - _Bug_Condition: Prevents unclipped scale from being written to size_map when scale is out of range_
    - _Expected_Behavior: size_map stores the same scale value used for LUT building during pointer creation_
    - _Preservation: In-range scales pass through unchanged (clipped == unclipped)_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.3 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Out-of-Range Scale Pointer Validity
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - Verify that out-of-range scales now produce valid pointer indices
    - Verify that clipped scale is returned and used consistently
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.4 Verify preservation tests still pass
    - **Property 2: Preservation** - In-Range Scale Behavior Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after fix (no regressions for in-range scales)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [ ] 4. Run integration tests to verify 100% MATLAB parity

  - [x] 4.1 Run parity experiment preflight check
    - Run: `python dev/scripts/cli/parity_experiment.py preflight-exact --source-run-root <source> --dest-run-root <dest>`
    - Verify preflight passes (staged artifacts are ready for comparison)
    - _Requirements: 2.5_

  - [x] 4.2 Verify LUT building is unchanged
    - Run: `python dev/scripts/cli/parity_experiment.py prove-luts --source-run-root <source> --dest-run-root <dest>`
    - Verify LUT building produces identical results for all scale indices
    - _Requirements: 3.5_

  - [x] 4.3 Verify candidate capture achieves 100% match rate
    - Run: `python dev/scripts/cli/parity_experiment.py capture-candidates --source-run-root <source> --dest-run-root <dest>`
    - Verify 100% candidate match rate with MATLAB (2533 candidates)
    - Verify zero missing candidates (was 890 on unfixed code)
    - Verify zero extra candidates (was 477 on unfixed code)
    - Verify zero pointer out-of-range errors in logs
    - _Requirements: 2.3, 2.4, 2.5_

  - [~] 4.4 Verify edge replay is unchanged
    - Run: `python dev/scripts/cli/parity_experiment.py replay-edges --source-run-root <source> --dest-run-root <dest>`
    - Verify edge tracing logic produces identical results
    - _Requirements: 3.6, 3.7_

  - [~] 4.5 Verify exact parity with MATLAB vectors
    - Run: `python dev/scripts/cli/parity_experiment.py prove-exact --source-run-root <source> --dest-run-root <dest> --stage all`
    - Verify all normalized checkpoints match MATLAB vectors
    - _Requirements: 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [~] 5. Checkpoint - Ensure all tests pass
  - Verify all property-based tests pass (bug condition and preservation)
  - Verify all integration tests pass (100% MATLAB parity achieved)
  - Verify zero pointer out-of-range errors in diagnostic logs
  - Ask the user if questions arise
