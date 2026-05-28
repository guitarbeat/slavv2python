"""CI integration test for the paper-profile pipeline.

Verifies that the 'paper' profile works correctly on synthetic data,
ensuring regression tests can run in environments without real datasets.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import tifffile

from slavv_python.engine import SlavvPipeline
from slavv_python.storage import load_tiff_volume
from slavv_python.utils.synthetic import (
    generate_synthetic_vessel_volume,
    generate_synthetic_y_junction_volume,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_paper_profile_ci_pipeline(tmp_path: Path):
    """Run the pipeline with 'paper' profile on a synthetic volume."""

    # 1. Generate and save a synthetic volume
    volume_path = tmp_path / "synthetic_volume.tif"
    synthetic_image = generate_synthetic_vessel_volume(
        shape=(32, 64, 64),
        vessel_radius=3.0,
        background_val=0.0,
        vessel_val=1.0,
    )
    tifffile.imwrite(str(volume_path), synthetic_image)

    # 2. Load the sample volume through the normal storage layer
    image = load_tiff_volume(str(volume_path))
    assert image is not None

    # 3. Setup run directory and parameters
    run_dir = tmp_path / "paper_run"
    params = {
        "pipeline_profile": "paper",
        # Speed up the test by reducing scales
        "radius_of_largest_vessel_in_microns": 5.0,
        "scales_per_octave": 1.0,
    }

    # 4. Run pipeline
    pipeline = SlavvPipeline()
    results = pipeline.run(
        image,
        params,
        run_dir=str(run_dir),
    )

    # 5. Verify results structure
    assert "vertices" in results
    assert "edges" in results
    assert "network" in results
    assert results["parameters"]["pipeline_profile"] == "paper"
    assert results["parameters"]["energy_projection_mode"] == "paper"

    # 6. Verify JSON export
    from slavv_python.storage import save_network_to_json

    json_path = run_dir / "network.json"
    save_network_to_json(results, str(json_path))
    assert json_path.exists()

    with open(json_path) as f:
        data = json.load(f)

    assert data["schema"]["name"] == "slavv-network"
    assert data["metadata"]["pipeline_profile"] == "paper"
    assert "vertices" in data
    assert "edges" in data
    assert "network" in data

    # 7. Verify network has data
    assert len(data["network"]["strands"]) > 0


def test_paper_profile_ci_pipeline_on_y_junction_synthetic(tmp_path: Path):
    """Run paper profile on a non-trivial synthetic topology (parity pre-gate tier 1)."""
    volume_path = tmp_path / "synthetic_y_junction.tif"
    synthetic_image = generate_synthetic_y_junction_volume(
        shape=(32, 64, 64),
        trunk_radius=3.0,
        branch_radius=3.0,
    )
    tifffile.imwrite(str(volume_path), synthetic_image)

    image = load_tiff_volume(str(volume_path))
    run_dir = tmp_path / "paper_run_y"
    params = {
        "pipeline_profile": "paper",
        "radius_of_largest_vessel_in_microns": 5.0,
        "scales_per_octave": 1.0,
    }

    results = SlavvPipeline().run(image, params, run_dir=str(run_dir))
    assert len(results["network"]["strands"]) > 0
