"""Parity Pre-Gate tier 1 — synthetic fixture smoke (ADR 0009).

Tier 1 validates pipeline and parity seams on Python-generated volumes without
MATLAB oracles or ``prove-exact`` certification claims.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from slavv_python.analytics.parity.matlab_fail_fast import compare_lut_fixture_payload
from slavv_python.engine import SlavvPipeline
from slavv_python.processing.stages.edges.watershed_lut import build_matlab_global_watershed_lut
from slavv_python.storage import load_tiff_volume
from slavv_python.utils.synthetic import generate_synthetic_y_junction_volume

if TYPE_CHECKING:
    from pathlib import Path


def _lut_fixture_payload() -> dict[str, object]:
    lut = build_matlab_global_watershed_lut(
        0,
        size_of_image=(5, 5, 5),
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        step_size_per_origin_radius=1.0,
    )
    return {
        "size_of_image": [5, 5, 5],
        "microns_per_voxel": [1.0, 1.0, 1.0],
        "lumen_radius_microns": [1.0],
        "scales": {
            "0": {
                "linear_offsets": np.asarray(lut["linear_offsets"], dtype=np.int64).tolist(),
                "local_subscripts": np.asarray(lut["local_subscripts"], dtype=np.int32).tolist(),
                "r_over_R": np.asarray(lut["r_over_R"], dtype=np.float32).tolist(),
                "unit_vectors": np.asarray(lut["unit_vectors"], dtype=np.float32).tolist(),
            }
        },
    }


@pytest.mark.integration
@pytest.mark.parity
def test_tier1_watershed_lut_seam_matches_parity_comparator():
    """Watershed LUT module and parity fail-fast comparator share one deep seam."""
    fixture = _lut_fixture_payload()
    report = compare_lut_fixture_payload(
        fixture,
        size_of_image=(5, 5, 5),
        microns_per_voxel=np.ones((3,), dtype=np.float32),
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
    )
    assert report["passed"] is True
    assert report["first_failure"] is None


@pytest.mark.integration
@pytest.mark.parity
def test_tier1_matlab_compat_synthetic_edges_smoke(tmp_path: Path):
    """Run matlab_compat through Edge Discovery on a Y-junction Synthetic Fixture Volume."""
    volume_path = tmp_path / "synthetic_y_junction.tif"
    synthetic_image = generate_synthetic_y_junction_volume(
        shape=(32, 64, 64),
        trunk_radius=3.0,
        branch_radius=3.0,
    )
    tifffile.imwrite(str(volume_path), synthetic_image)

    image = load_tiff_volume(str(volume_path))
    run_dir = tmp_path / "matlab_compat_run"
    params = {
        "pipeline_profile": "matlab_compat",
        "radius_of_largest_vessel_in_microns": 5.0,
        "scales_per_octave": 1.0,
    }

    results = SlavvPipeline().run(
        image,
        params,
        run_dir=str(run_dir),
        stop_after="edges",
    )

    assert results["parameters"]["pipeline_profile"] == "matlab_compat"
    assert results["parameters"]["energy_projection_mode"] == "matlab"
    assert len(results["vertices"]["positions"]) > 0
    assert results["edges"]["connections"].size > 0
