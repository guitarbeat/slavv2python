"""Tests for synthetic vessel volume generators and ladder rung registry."""

from __future__ import annotations

import numpy as np
import pytest

from slavv_python.utils.synthetic import (
    LADDER_RUNG_IDS,
    LADDER_RUNG_MAX_DIM,
    generate_ladder_rung_volume,
    generate_synthetic_asymmetric_y_volume,
    generate_synthetic_double_junction_volume,
    generate_synthetic_vessel_volume,
    generate_synthetic_y_junction_volume,
)


@pytest.mark.unit
def test_y_junction_has_branch_voxels_beyond_trunk_cross_section():
    shape = (32, 64, 64)
    trunk = generate_synthetic_vessel_volume(shape=shape, vessel_radius=3.0, vessel_val=1.0)
    junction = generate_synthetic_y_junction_volume(
        shape=shape, trunk_radius=3.0, branch_radius=3.0, vessel_val=1.0
    )

    assert junction.shape == shape
    assert junction.sum() > trunk.sum()
    cy, _cx = shape[1] // 2, shape[2] // 2
    z_mid = shape[0] // 2
    assert junction[z_mid, cy, -1] == pytest.approx(1.0)
    assert trunk[z_mid, cy, -1] == pytest.approx(0.0)


@pytest.mark.unit
def test_ladder_rung_shapes_and_vessel_presence():
    expected_shapes = {
        "y_junction_32": (32, 32, 32),
        "double_junction_32": (32, 32, 32),
        "asymmetric_y_48": (48, 48, 48),
        "y_junction_64": (64, 64, 64),
    }
    for rung_id in LADDER_RUNG_IDS:
        vol = generate_ladder_rung_volume(rung_id)
        assert vol.shape == expected_shapes[rung_id]
        assert int((vol > 0.5).sum()) > 0
        assert LADDER_RUNG_MAX_DIM[rung_id] == max(vol.shape)


@pytest.mark.unit
def test_ladder_rung1_matches_tiny_experiment_y_junction_call():
    """Baseline rung matches the tiny experiment's 32³ Y-junction paint params."""
    baseline = generate_ladder_rung_volume("y_junction_32")
    explicit = generate_synthetic_y_junction_volume(
        shape=(32, 32, 32),
        trunk_radius=3.0,
        branch_radius=3.0,
        background_val=0.0,
        vessel_val=1.0,
    )
    assert baseline.shape == (32, 32, 32)
    assert np.array_equal(baseline, explicit)


@pytest.mark.unit
def test_double_junction_has_topology_discriminator_vs_single_y():
    single = generate_synthetic_y_junction_volume(
        shape=(32, 32, 32), trunk_radius=3.0, branch_radius=3.0, vessel_val=1.0
    )
    double = generate_synthetic_double_junction_volume(
        shape=(32, 32, 32), trunk_radius=3.0, branch_radius=3.0, vessel_val=1.0
    )
    assert double.shape == single.shape
    assert int((double > 0.5).sum()) > int((single > 0.5).sum())
    # Second branch paints toward -X at a lower Z plane.
    cy = 16
    z2 = 8
    assert double[z2, cy, 0] == pytest.approx(1.0)
    assert single[z2, cy, 0] == pytest.approx(0.0)


@pytest.mark.unit
def test_asymmetric_y_has_offset_junction():
    vol = generate_synthetic_asymmetric_y_volume(
        shape=(48, 48, 48),
        trunk_radius=4.0,
        branch_radius=2.5,
        junction_y_offset=6,
        vessel_val=1.0,
    )
    assert vol.shape == (48, 48, 48)
    cy = 48 // 2 + 6
    cx = 48 // 2
    z_mid = 24
    assert vol[z_mid, cy, cx] == pytest.approx(1.0)
    # Center-of-volume column should not be the trunk when offset is large.
    assert vol[z_mid, 24, cx] == pytest.approx(0.0)


@pytest.mark.unit
def test_ladder_rung_determinism():
    a = generate_ladder_rung_volume("asymmetric_y_48")
    b = generate_ladder_rung_volume("asymmetric_y_48")
    assert np.array_equal(a, b)


@pytest.mark.unit
def test_unknown_ladder_rung_raises():
    with pytest.raises(ValueError, match="unknown ladder rung"):
        generate_ladder_rung_volume("not_a_rung")
