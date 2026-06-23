"""Tests for synthetic vessel volume generators."""

from __future__ import annotations

import pytest

from slavv_python.utils.synthetic import (
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
