"""Integration test for the paper-profile pipeline.

Verifies that the 'paper' profile defaults work correctly and produce
the expected output artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from slavv_python.core import SlavvPipeline
from slavv_python.io import load_tiff_volume

DATA_DIR = Path(__file__).resolve().parents[2] / "workspace" / "datasets"
TEST_VOLUME = DATA_DIR / "180709_EL_center_crop_24x256x256.tif"


@pytest.mark.skipif(not TEST_VOLUME.exists(), reason="Sample dataset not available")
def test_paper_profile_pipeline_integration(tmp_path):
    """Run the pipeline with 'paper' profile and verify JSON export."""
    
    # 1. Load sample volume
    image = load_tiff_volume(str(TEST_VOLUME))
    assert image is not None
    
    # 2. Setup run directory and parameters
    run_dir = tmp_path / "paper_run"
    params = {
        "pipeline_profile": "paper",
        # Speed up the test by reducing scales
        "radius_of_largest_vessel_in_microns": 5.0,
        "scales_per_octave": 1.0,
    }
    
    # 3. Run pipeline
    pipeline = SlavvPipeline()
    results = pipeline.run(
        image,
        params,
        run_dir=str(run_dir),
    )
    
    # 4. Verify results structure
    assert "vertices" in results
    assert "edges" in results
    assert "network" in results
    assert results["parameters"]["pipeline_profile"] == "paper"
    assert results["parameters"]["energy_projection_mode"] == "paper"
    
    # 5. Verify JSON export
    from slavv_python.io import save_network_to_json
    json_path = run_dir / "network.json"
    save_network_to_json(results, str(json_path))
    assert json_path.exists()
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    assert data["schema"]["name"] == "slavv-network"
    assert data["metadata"]["pipeline_profile"] == "paper"
    assert "vertices" in data
    assert "edges" in data
    assert "network" in data
    
    # 6. Verify network has data
    assert len(data["network"]["strands"]) > 0
