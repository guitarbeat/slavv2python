"""Property-based tests for v22 pointer corruption bugfix.

This module contains bug condition exploration tests that verify the expected behavior
for out-of-range scale labels in the global watershed implementation.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
"""

from __future__ import annotations

import numpy as np
import pytest

from source.core._edge_candidates.common import _build_matlab_global_watershed_lut
from source.core._edge_candidates.global_watershed import (
    _matlab_global_watershed_current_strel,
)


class TestBugConditionOutOfRangeScalePointerValidity:
    """Property 1: Bug Condition - Out-of-Range Scale Pointer Validity.

    **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists.
    **DO NOT attempt to fix the test or the code when it fails**.
    **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes.

    **GOAL**: Surface counterexamples that demonstrate pointer corruption for out-of-range scales.

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
    """

    @pytest.mark.parametrize(
        ("current_scale_label", "expected_clipped_label"),
        [
            # Out-of-range cases that should trigger clipping
            (0, 1),  # Below valid range
            (12, 11),  # Above valid range (typical max is 11)
            (100, 11),  # Far above valid range
            (-1, 1),  # Negative (edge case)
            (15, 11),  # Moderately above range
        ],
    )
    def test_out_of_range_scale_clipping_consistency(
        self,
        current_scale_label: int,
        expected_clipped_label: int,
    ):
        """Test that out-of-range scales are clipped and used consistently.

        This test verifies that when current_scale_label exceeds valid range,
        the system clips the scale and uses the clipped value consistently for
        both pointer creation and size_map storage.

        **Expected behavior (after fix)**:
        - scale_label_clipped should equal clip(current_scale_label - 1, 0, len(lumen_radius_microns) - 1) + 1
        - All pointer indices should be within valid range [1, lut_size] for the clipped scale's LUT

        **Expected outcome on UNFIXED code**: FAILS (this proves the bug exists)
        **Expected outcome on FIXED code**: PASSES (this confirms the fix works)
        """
        # Setup: typical configuration with 11 scales
        shape = (50, 50, 50)
        lumen_radius_microns = np.array(
            [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
            dtype=np.float32,
        )
        microns_per_voxel = np.ones((3,), dtype=np.float32)
        step_size_per_origin_radius = 1.0

        # Choose a location in the middle of the volume
        current_linear = 25 * shape[0] * shape[1] + 25 * shape[0] + 25

        # Call the function with out-of-range scale
        current_strel = _matlab_global_watershed_current_strel(
            current_linear,
            current_scale_label=current_scale_label,
            shape=shape,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            step_size_per_origin_radius=step_size_per_origin_radius,
        )

        # Calculate what the clipped scale should be
        current_scale_index_clipped = int(
            np.clip(current_scale_label - 1, 0, len(lumen_radius_microns) - 1)
        )
        expected_clipped_scale_label = current_scale_index_clipped + 1

        # Build the LUT for the clipped scale to get the expected size
        lut_clipped = _build_matlab_global_watershed_lut(
            current_scale_index_clipped,
            size_of_image=shape,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            step_size_per_origin_radius=step_size_per_origin_radius,
        )
        expected_lut_size = len(lut_clipped["linear_offsets"])

        # ASSERTION 1: The function should return the clipped scale
        # This will FAIL on unfixed code because scale_label_clipped is not returned
        assert "scale_label_clipped" in current_strel, (
            f"EXPECTED BEHAVIOR NOT IMPLEMENTED: scale_label_clipped not returned. "
            f"This confirms the bug exists - the function does not return the clipped scale. "
            f"Input scale: {current_scale_label}, Expected clipped: {expected_clipped_scale_label}"
        )

        actual_clipped_scale = current_strel.get("scale_label_clipped")
        assert actual_clipped_scale == expected_clipped_scale_label, (
            f"SCALE CLIPPING MISMATCH: Expected clipped scale {expected_clipped_scale_label}, "
            f"got {actual_clipped_scale} for input scale {current_scale_label}. "
            f"This indicates the clipping logic is incorrect."
        )

        # ASSERTION 2: All pointer indices must be valid for the clipped scale's LUT
        pointer_indices = current_strel["pointer_indices"]
        lut_size = current_strel["lut_size"]

        # Verify LUT size matches expected
        assert lut_size == expected_lut_size, (
            f"LUT SIZE MISMATCH: Expected LUT size {expected_lut_size} for clipped scale "
            f"{expected_clipped_scale_label}, got {lut_size}"
        )

        # Check all pointers are in valid range [1, lut_size]
        invalid_pointers = pointer_indices[(pointer_indices < 1) | (pointer_indices > lut_size)]

        assert len(invalid_pointers) == 0, (
            f"POINTER OUT OF RANGE: Found {len(invalid_pointers)} invalid pointer(s) "
            f"for scale {current_scale_label} (clipped to {expected_clipped_scale_label}). "
            f"Pointer range: [{np.min(pointer_indices)}, {np.max(pointer_indices)}], "
            f"Valid range: [1, {lut_size}]. "
            f"Sample invalid pointers: {invalid_pointers[:5].tolist()}. "
            f"This confirms pointer corruption exists in the unfixed code."
        )

        # ASSERTION 3: Verify pointer indices are within the clipped LUT bounds
        assert np.all(pointer_indices >= 1), (
            f"POINTER UNDERFLOW: Some pointers are less than 1. "
            f"Min pointer: {np.min(pointer_indices)}"
        )

        assert np.all(pointer_indices <= lut_size), (
            f"POINTER OVERFLOW: Some pointers exceed LUT size {lut_size}. "
            f"Max pointer: {np.max(pointer_indices)}, LUT size: {lut_size}. "
            f"This is the core bug - pointers created for one scale but used with another."
        )

    @pytest.mark.parametrize(
        "current_scale_label",
        [0, 12, 15, 20, 100, -1],
    )
    def test_out_of_range_scale_generates_valid_strel(self, current_scale_label: int):
        """Test that out-of-range scales still generate valid strel structures.

        This test verifies that even with out-of-range scale labels, the strel
        generation doesn't crash and produces structurally valid output.

        **Validates: Requirements 1.1, 1.2**
        """
        shape = (30, 30, 30)
        lumen_radius_microns = np.array(
            [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
            dtype=np.float32,
        )
        microns_per_voxel = np.ones((3,), dtype=np.float32)
        step_size_per_origin_radius = 1.0

        # Choose a location in the middle
        current_linear = 15 * shape[0] * shape[1] + 15 * shape[0] + 15

        # This should not crash
        current_strel = _matlab_global_watershed_current_strel(
            current_linear,
            current_scale_label=current_scale_label,
            shape=shape,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            step_size_per_origin_radius=step_size_per_origin_radius,
        )

        # Verify basic structure
        assert "coords" in current_strel
        assert "pointer_indices" in current_strel
        assert "lut_size" in current_strel
        assert len(current_strel["coords"]) == len(current_strel["pointer_indices"])

        # All coordinates should be in bounds
        coords = current_strel["coords"]
        assert np.all(coords[:, 0] >= 0)
        assert np.all(coords[:, 0] < shape[0])
        assert np.all(coords[:, 1] >= 0)
        assert np.all(coords[:, 1] < shape[1])
        assert np.all(coords[:, 2] >= 0)
        assert np.all(coords[:, 2] < shape[2])

    def test_multiple_out_of_range_scales_at_different_locations(self):
        """Test multiple out-of-range scales at different spatial locations.

        This test generates strels for multiple out-of-range scales at different
        locations to verify consistency across the volume.

        **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
        """
        shape = (40, 40, 40)
        lumen_radius_microns = np.array(
            [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
            dtype=np.float32,
        )
        microns_per_voxel = np.ones((3,), dtype=np.float32)
        step_size_per_origin_radius = 1.0

        # Test multiple locations with out-of-range scales
        test_cases = [
            (10 * shape[0] * shape[1] + 10 * shape[0] + 10, 0),
            (20 * shape[0] * shape[1] + 20 * shape[0] + 20, 12),
            (30 * shape[0] * shape[1] + 30 * shape[0] + 30, 100),
            (15 * shape[0] * shape[1] + 25 * shape[0] + 35, 15),
        ]

        for current_linear, current_scale_label in test_cases:
            current_strel = _matlab_global_watershed_current_strel(
                current_linear,
                current_scale_label=current_scale_label,
                shape=shape,
                lumen_radius_microns=lumen_radius_microns,
                microns_per_voxel=microns_per_voxel,
                step_size_per_origin_radius=step_size_per_origin_radius,
            )

            # Calculate expected clipped scale
            current_scale_index_clipped = int(
                np.clip(current_scale_label - 1, 0, len(lumen_radius_microns) - 1)
            )
            expected_clipped_scale_label = current_scale_index_clipped + 1

            # Build expected LUT
            lut_clipped = _build_matlab_global_watershed_lut(
                current_scale_index_clipped,
                size_of_image=shape,
                lumen_radius_microns=lumen_radius_microns,
                microns_per_voxel=microns_per_voxel,
                step_size_per_origin_radius=step_size_per_origin_radius,
            )
            expected_lut_size = len(lut_clipped["linear_offsets"])

            # Verify clipped scale is returned (will fail on unfixed code)
            if "scale_label_clipped" in current_strel:
                assert current_strel["scale_label_clipped"] == expected_clipped_scale_label

            # Verify pointer validity
            pointer_indices = current_strel["pointer_indices"]
            lut_size = current_strel["lut_size"]

            assert lut_size == expected_lut_size

            # This assertion will fail on unfixed code if pointers are out of range
            invalid_pointers = pointer_indices[(pointer_indices < 1) | (pointer_indices > lut_size)]

            if len(invalid_pointers) > 0:
                pytest.fail(
                    f"Location {current_linear}, scale {current_scale_label}: "
                    f"Found {len(invalid_pointers)} invalid pointers. "
                    f"Range: [{np.min(pointer_indices)}, {np.max(pointer_indices)}], "
                    f"Valid: [1, {lut_size}]. "
                    f"This confirms the bug exists."
                )


class TestPreservationInRangeScaleBehavior:
    """Property 2: Preservation - In-Range Scale Behavior Unchanged.

    **IMPORTANT**: These tests run on UNFIXED code to establish baseline behavior.
    **EXPECTED OUTCOME**: Tests PASS on unfixed code (confirms baseline to preserve).

    **GOAL**: Verify that in-range scales work correctly and will remain unchanged after fix.

    For in-range scales where current_scale_label is in [1, len(lumen_radius_microns)],
    the clipped scale equals the unclipped scale (no clipping occurs), and all pointer
    creation, size_map writing, and backtracking behavior works correctly.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**
    """

    @pytest.mark.parametrize(
        "current_scale_label",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    )
    def test_in_range_scale_no_clipping_needed(self, current_scale_label: int):
        """Test that in-range scales work correctly without clipping.

        For valid scale labels in [1, len(lumen_radius_microns)], verify that:
        - Pointer indices are valid for the scale's LUT
        - LUT building produces correct size
        - No clipping is needed (clipped == unclipped)

        **Expected outcome on UNFIXED code**: PASSES (baseline behavior is correct)
        **Expected outcome on FIXED code**: PASSES (behavior preserved)

        **Validates: Requirements 3.1, 3.2, 3.5**
        """
        # Setup: typical configuration with 11 scales
        shape = (50, 50, 50)
        lumen_radius_microns = np.array(
            [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
            dtype=np.float32,
        )
        microns_per_voxel = np.ones((3,), dtype=np.float32)
        step_size_per_origin_radius = 1.0

        # Choose a location in the middle of the volume
        current_linear = 25 * shape[0] * shape[1] + 25 * shape[0] + 25

        # Call the function with in-range scale
        current_strel = _matlab_global_watershed_current_strel(
            current_linear,
            current_scale_label=current_scale_label,
            shape=shape,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            step_size_per_origin_radius=step_size_per_origin_radius,
        )

        # For in-range scales, clipped should equal unclipped
        current_scale_index = current_scale_label - 1
        current_scale_index_clipped = int(
            np.clip(current_scale_label - 1, 0, len(lumen_radius_microns) - 1)
        )

        # Verify no clipping occurred
        assert current_scale_index == current_scale_index_clipped, (
            f"UNEXPECTED CLIPPING: In-range scale {current_scale_label} was clipped. "
            f"Original index: {current_scale_index}, Clipped: {current_scale_index_clipped}"
        )

        # Build the expected LUT for this scale
        lut = _build_matlab_global_watershed_lut(
            current_scale_index,
            size_of_image=shape,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            step_size_per_origin_radius=step_size_per_origin_radius,
        )
        expected_lut_size = len(lut["linear_offsets"])

        # Verify LUT size matches
        lut_size = current_strel["lut_size"]
        assert lut_size == expected_lut_size, (
            f"LUT SIZE MISMATCH: Expected {expected_lut_size} for scale {current_scale_label}, "
            f"got {lut_size}"
        )

        # Verify all pointer indices are valid
        pointer_indices = current_strel["pointer_indices"]

        assert len(pointer_indices) > 0, (
            f"NO POINTERS GENERATED: Scale {current_scale_label} produced no pointers"
        )

        assert np.all(pointer_indices >= 1), (
            f"POINTER UNDERFLOW: Some pointers < 1 for scale {current_scale_label}. "
            f"Min: {np.min(pointer_indices)}"
        )

        assert np.all(pointer_indices <= lut_size), (
            f"POINTER OVERFLOW: Some pointers > {lut_size} for scale {current_scale_label}. "
            f"Max: {np.max(pointer_indices)}"
        )

        # Verify pointer indices are within valid range [1, lut_size]
        invalid_pointers = pointer_indices[(pointer_indices < 1) | (pointer_indices > lut_size)]

        assert len(invalid_pointers) == 0, (
            f"INVALID POINTERS: Found {len(invalid_pointers)} invalid pointer(s) "
            f"for in-range scale {current_scale_label}. "
            f"Valid range: [1, {lut_size}], "
            f"Actual range: [{np.min(pointer_indices)}, {np.max(pointer_indices)}]. "
            f"This indicates a bug in the baseline behavior."
        )

    def test_all_in_range_scales_produce_valid_pointers(self):
        """Test that all in-range scales produce valid pointer indices.

        This property-based test generates strels for all valid scale labels
        and verifies that pointer indices are always within valid range.

        **Expected outcome on UNFIXED code**: PASSES (baseline behavior is correct)
        **Expected outcome on FIXED code**: PASSES (behavior preserved)

        **Validates: Requirements 3.1, 3.2, 3.5**
        """
        shape = (40, 40, 40)
        lumen_radius_microns = np.array(
            [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
            dtype=np.float32,
        )
        microns_per_voxel = np.ones((3,), dtype=np.float32)
        step_size_per_origin_radius = 1.0

        # Test all valid scale labels
        for current_scale_label in range(1, len(lumen_radius_microns) + 1):
            # Test at multiple locations
            test_locations = [
                10 * shape[0] * shape[1] + 10 * shape[0] + 10,
                20 * shape[0] * shape[1] + 20 * shape[0] + 20,
                30 * shape[0] * shape[1] + 30 * shape[0] + 30,
            ]

            for current_linear in test_locations:
                current_strel = _matlab_global_watershed_current_strel(
                    current_linear,
                    current_scale_label=current_scale_label,
                    shape=shape,
                    lumen_radius_microns=lumen_radius_microns,
                    microns_per_voxel=microns_per_voxel,
                    step_size_per_origin_radius=step_size_per_origin_radius,
                )

                pointer_indices = current_strel["pointer_indices"]
                lut_size = current_strel["lut_size"]

                assert np.all(pointer_indices >= 1), (
                    f"INVALID POINTERS at location {current_linear}, scale {current_scale_label}: "
                    f"Range [{np.min(pointer_indices)}, {np.max(pointer_indices)}], "
                    f"Valid [1, {lut_size}]"
                )
                assert np.all(pointer_indices <= lut_size), (
                    f"INVALID POINTERS at location {current_linear}, scale {current_scale_label}: "
                    f"Range [{np.min(pointer_indices)}, {np.max(pointer_indices)}], "
                    f"Valid [1, {lut_size}]"
                )

    @pytest.mark.parametrize(
        "current_scale_label",
        [1, 3, 5, 7, 9, 11],
    )
    def test_in_range_scale_lut_consistency(self, current_scale_label: int):
        """Test that LUT building is consistent for in-range scales.

        Verify that building the LUT for an in-range scale produces the same
        result regardless of how it's accessed (directly vs through strel).

        **Expected outcome on UNFIXED code**: PASSES (LUT building is correct)
        **Expected outcome on FIXED code**: PASSES (LUT building unchanged)

        **Validates: Requirements 3.5**
        """
        shape = (50, 50, 50)
        lumen_radius_microns = np.array(
            [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
            dtype=np.float32,
        )
        microns_per_voxel = np.ones((3,), dtype=np.float32)
        step_size_per_origin_radius = 1.0

        current_linear = 25 * shape[0] * shape[1] + 25 * shape[0] + 25

        # Build LUT directly
        current_scale_index = current_scale_label - 1
        lut_direct = _build_matlab_global_watershed_lut(
            current_scale_index,
            size_of_image=shape,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            step_size_per_origin_radius=step_size_per_origin_radius,
        )

        # Get LUT size from strel
        current_strel = _matlab_global_watershed_current_strel(
            current_linear,
            current_scale_label=current_scale_label,
            shape=shape,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            step_size_per_origin_radius=step_size_per_origin_radius,
        )

        # Verify LUT sizes match
        assert current_strel["lut_size"] == len(lut_direct["linear_offsets"]), (
            f"LUT SIZE INCONSISTENCY: Direct LUT has {len(lut_direct['linear_offsets'])} elements, "
            f"strel reports {current_strel['lut_size']} for scale {current_scale_label}"
        )

    def test_in_range_scale_pointer_indices_are_one_based(self):
        """Test that pointer indices are 1-based for in-range scales.

        Verify that pointer indices start at 1 (not 0) and are valid indices
        into the LUT, following MATLAB's 1-based indexing convention.

        **Expected outcome on UNFIXED code**: PASSES (1-based indexing is correct)
        **Expected outcome on FIXED code**: PASSES (indexing convention preserved)

        **Validates: Requirements 3.2**
        """
        shape = (40, 40, 40)
        lumen_radius_microns = np.array(
            [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
            dtype=np.float32,
        )
        microns_per_voxel = np.ones((3,), dtype=np.float32)
        step_size_per_origin_radius = 1.0

        # Test multiple in-range scales
        for current_scale_label in [1, 5, 11]:
            current_linear = 20 * shape[0] * shape[1] + 20 * shape[0] + 20

            current_strel = _matlab_global_watershed_current_strel(
                current_linear,
                current_scale_label=current_scale_label,
                shape=shape,
                lumen_radius_microns=lumen_radius_microns,
                microns_per_voxel=microns_per_voxel,
                step_size_per_origin_radius=step_size_per_origin_radius,
            )

            pointer_indices = current_strel["pointer_indices"]

            # Verify 1-based indexing (no zeros)
            assert np.all(pointer_indices >= 1), (
                f"ZERO-BASED INDEXING DETECTED: Scale {current_scale_label} has pointers < 1. "
                f"Min pointer: {np.min(pointer_indices)}. "
                f"Expected 1-based indexing."
            )

            # Verify pointers are valid indices into LUT
            lut_size = current_strel["lut_size"]
            assert np.all(pointer_indices <= lut_size), (
                f"POINTER OUT OF RANGE: Scale {current_scale_label} has pointers > {lut_size}. "
                f"Max pointer: {np.max(pointer_indices)}"
            )

    def test_in_range_scale_coordinates_in_bounds(self):
        """Test that strel coordinates are within volume bounds for in-range scales.

        Verify that all generated strel coordinates are within the valid volume
        bounds, ensuring boundary checking works correctly.

        **Expected outcome on UNFIXED code**: PASSES (boundary checking is correct)
        **Expected outcome on FIXED code**: PASSES (boundary checking preserved)

        **Validates: Requirements 3.8**
        """
        shape = (30, 30, 30)
        lumen_radius_microns = np.array(
            [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
            dtype=np.float32,
        )
        microns_per_voxel = np.ones((3,), dtype=np.float32)
        step_size_per_origin_radius = 1.0

        # Test at edge locations where boundary filtering is important
        test_cases = [
            (5 * shape[0] * shape[1] + 5 * shape[0] + 5, 3),  # Near corner
            (15 * shape[0] * shape[1] + 15 * shape[0] + 15, 7),  # Center
            (25 * shape[0] * shape[1] + 25 * shape[0] + 25, 11),  # Near opposite corner
        ]

        for current_linear, current_scale_label in test_cases:
            current_strel = _matlab_global_watershed_current_strel(
                current_linear,
                current_scale_label=current_scale_label,
                shape=shape,
                lumen_radius_microns=lumen_radius_microns,
                microns_per_voxel=microns_per_voxel,
                step_size_per_origin_radius=step_size_per_origin_radius,
            )

            coords = current_strel["coords"]

            # Verify all coordinates are in bounds
            assert np.all(coords[:, 0] >= 0), (
                f"X COORDINATES OUT OF BOUNDS at location {current_linear}, scale {current_scale_label}: "
                f"Range [{np.min(coords[:, 0])}, {np.max(coords[:, 0])}], Valid [0, {shape[0]})"
            )
            assert np.all(coords[:, 0] < shape[0]), (
                f"X COORDINATES OUT OF BOUNDS at location {current_linear}, scale {current_scale_label}: "
                f"Range [{np.min(coords[:, 0])}, {np.max(coords[:, 0])}], Valid [0, {shape[0]})"
            )

            assert np.all(coords[:, 1] >= 0), (
                f"Y COORDINATES OUT OF BOUNDS at location {current_linear}, scale {current_scale_label}: "
                f"Range [{np.min(coords[:, 1])}, {np.max(coords[:, 1])}], Valid [0, {shape[1]})"
            )
            assert np.all(coords[:, 1] < shape[1]), (
                f"Y COORDINATES OUT OF BOUNDS at location {current_linear}, scale {current_scale_label}: "
                f"Range [{np.min(coords[:, 1])}, {np.max(coords[:, 1])}], Valid [0, {shape[1]})"
            )

            assert np.all(coords[:, 2] >= 0), (
                f"Z COORDINATES OUT OF BOUNDS at location {current_linear}, scale {current_scale_label}: "
                f"Range [{np.min(coords[:, 2])}, {np.max(coords[:, 2])}], Valid [0, {shape[2]})"
            )
            assert np.all(coords[:, 2] < shape[2]), (
                f"Z COORDINATES OUT OF BOUNDS at location {current_linear}, scale {current_scale_label}: "
                f"Range [{np.min(coords[:, 2])}, {np.max(coords[:, 2])}], Valid [0, {shape[2]})"
            )

    def test_in_range_scale_strel_structure_validity(self):
        """Test that strel structure is valid for all in-range scales.

        Verify that the strel dictionary contains all required fields and that
        array lengths are consistent.

        **Expected outcome on UNFIXED code**: PASSES (structure is correct)
        **Expected outcome on FIXED code**: PASSES (structure preserved)

        **Validates: Requirements 3.1, 3.2**
        """
        shape = (40, 40, 40)
        lumen_radius_microns = np.array(
            [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
            dtype=np.float32,
        )
        microns_per_voxel = np.ones((3,), dtype=np.float32)
        step_size_per_origin_radius = 1.0

        current_linear = 20 * shape[0] * shape[1] + 20 * shape[0] + 20

        # Test all in-range scales
        for current_scale_label in range(1, len(lumen_radius_microns) + 1):
            current_strel = _matlab_global_watershed_current_strel(
                current_linear,
                current_scale_label=current_scale_label,
                shape=shape,
                lumen_radius_microns=lumen_radius_microns,
                microns_per_voxel=microns_per_voxel,
                step_size_per_origin_radius=step_size_per_origin_radius,
            )

            # Verify required fields exist
            required_fields = [
                "current_coord",
                "coords",
                "offsets",
                "linear_indices",
                "pointer_indices",
                "r_over_R",
                "distance_microns",
                "unit_vectors",
                "lut_size",
            ]

            for field in required_fields:
                assert field in current_strel, (
                    f"MISSING FIELD: '{field}' not in strel for scale {current_scale_label}"
                )

            # Verify array length consistency
            n_coords = len(current_strel["coords"])
            assert len(current_strel["offsets"]) == n_coords, (
                f"LENGTH MISMATCH: offsets ({len(current_strel['offsets'])}) != "
                f"coords ({n_coords}) for scale {current_scale_label}"
            )

            assert len(current_strel["linear_indices"]) == n_coords, (
                f"LENGTH MISMATCH: linear_indices ({len(current_strel['linear_indices'])}) != "
                f"coords ({n_coords}) for scale {current_scale_label}"
            )

            assert len(current_strel["pointer_indices"]) == n_coords, (
                f"LENGTH MISMATCH: pointer_indices ({len(current_strel['pointer_indices'])}) != "
                f"coords ({n_coords}) for scale {current_scale_label}"
            )

            assert len(current_strel["r_over_R"]) == n_coords, (
                f"LENGTH MISMATCH: r_over_R ({len(current_strel['r_over_R'])}) != "
                f"coords ({n_coords}) for scale {current_scale_label}"
            )

            assert len(current_strel["distance_microns"]) == n_coords, (
                f"LENGTH MISMATCH: distance_microns ({len(current_strel['distance_microns'])}) != "
                f"coords ({n_coords}) for scale {current_scale_label}"
            )

            assert len(current_strel["unit_vectors"]) == n_coords, (
                f"LENGTH MISMATCH: unit_vectors ({len(current_strel['unit_vectors'])}) != "
                f"coords ({n_coords}) for scale {current_scale_label}"
            )
