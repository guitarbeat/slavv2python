"""Property-based tests for structuring element float-radius membership.

# Feature: matlab-python-parity, Property 7: Structuring Element Float-Radius Membership

Verifies that ``ellipsoid_offsets`` with an isotropic radius ``r`` includes exactly
the voxel offsets ``(dy, dx, dz)`` satisfying ``sqrt(dy² + dx² + dz²) <= r`` using
float comparison, and excludes all offsets where the Euclidean distance strictly
exceeds ``r``.

The isotropic case (all three axis radii equal ``r``) reduces the anisotropic
membership criterion ``(dy/r0)² + (dx/r1)² + (dz/r2)² <= 1`` to the standard
Euclidean ball ``sqrt(dy² + dx² + dz²) <= r``, which is the property tested here.

Validates: Requirements 5.4
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slavv_python.pipeline.vertices.detection import ellipsoid_offsets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _euclidean_distance(dy: int, dx: int, dz: int) -> float:
    """Return the Euclidean distance of a voxel offset from the origin."""
    return math.sqrt(dy**2 + dx**2 + dz**2)


def _all_candidate_offsets(r: float) -> list[tuple[int, int, int]]:
    """Enumerate all integer offsets in the bounding cube [-ceil(r), +ceil(r)]^3."""
    bound = math.ceil(r)
    return [
        (dy, dx, dz)
        for dy in range(-bound, bound + 1)
        for dx in range(-bound, bound + 1)
        for dz in range(-bound, bound + 1)
    ]


# ---------------------------------------------------------------------------
# Property 7: Structuring Element Float-Radius Membership
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(r=st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None)
def test_included_offsets_satisfy_membership(r: float) -> None:
    """Every offset returned by ellipsoid_offsets satisfies sqrt(dy²+dx²+dz²) <= r.

    For an isotropic radius ``r`` (all axes equal), the general anisotropic
    membership test ``(dy/r)² + (dx/r)² + (dz/r)² <= 1`` is equivalent to
    ``sqrt(dy² + dx² + dz²) <= r``. This property checks that every voxel in the
    returned set satisfies the float-comparison membership criterion.
    """
    radii = np.array([r, r, r], dtype=np.float64)
    offsets = ellipsoid_offsets(radii)

    for row in offsets:
        dy, dx, dz = int(row[0]), int(row[1]), int(row[2])
        dist = _euclidean_distance(dy, dx, dz)
        assert dist <= r, (
            f"Included offset ({dy}, {dx}, {dz}) has distance {dist:.17g} > r={r:.17g}; "
            "it should not be in the returned set."
        )


@pytest.mark.unit
@given(r=st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None)
def test_excluded_offsets_do_not_satisfy_membership(r: float) -> None:
    """No integer offset with sqrt(dy²+dx²+dz²) > r appears in ellipsoid_offsets.

    Enumerate all integer offsets within [-ceil(r), +ceil(r)]^3, compute their
    Euclidean distance, and confirm that any offset whose distance strictly exceeds
    ``r`` is absent from the returned set.
    """
    radii = np.array([r, r, r], dtype=np.float64)
    offsets = ellipsoid_offsets(radii)

    # Build a set of returned offsets for O(1) membership testing.
    returned: set[tuple[int, int, int]] = {
        (int(row[0]), int(row[1]), int(row[2])) for row in offsets
    }

    for candidate in _all_candidate_offsets(r):
        dy, dx, dz = candidate
        dist = _euclidean_distance(dy, dx, dz)
        if dist > r:
            assert candidate not in returned, (
                f"Excluded offset ({dy}, {dx}, {dz}) with distance {dist:.17g} > r={r:.17g} "
                "was incorrectly included in the returned set."
            )


@pytest.mark.unit
@given(r=st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None)
def test_membership_criterion_is_exact_float_comparison(r: float) -> None:
    """ellipsoid_offsets uses float-radius membership, not integer-rounded radii.

    This test verifies the combined inclusion/exclusion property: the returned set
    is exactly ``{(dy, dx, dz) : sqrt(dy²+dx²+dz²) <= r}`` over the integer
    lattice within [-ceil(r), +ceil(r)]^3. Any deviation from this set (missing
    included offsets or spurious excluded offsets) would indicate that radii were
    rounded before the membership test rather than compared as floats.
    """
    radii = np.array([r, r, r], dtype=np.float64)
    offsets = ellipsoid_offsets(radii)

    returned: set[tuple[int, int, int]] = {
        (int(row[0]), int(row[1]), int(row[2])) for row in offsets
    }

    # Build the ground-truth set via direct float comparison.
    expected: set[tuple[int, int, int]] = {
        candidate
        for candidate in _all_candidate_offsets(r)
        if _euclidean_distance(*candidate) <= r
    }

    missing = expected - returned
    extra = returned - expected

    assert not missing and not extra, (
        f"For r={r:.17g}: "
        f"{len(missing)} missing offset(s): {sorted(missing)[:5]}; "
        f"{len(extra)} extra offset(s): {sorted(extra)[:5]}"
    )
