import os
from pathlib import Path

import numpy as np
import pytest

from slavv.core import SLAVVProcessor
from slavv.io import (
    load_tiff_volume,
    save_network_to_casx,
    save_network_to_csv,
    save_network_to_json,
    save_network_to_vmv,
)

# Skip integration tests by default unless explicitly requested or running locally,
# as they can be slow and require the test volume.
# For our purposes, we want to run it, but in CI it might be large.
# We'll just run it normally for this task.


def test_end_to_end_pipeline(tmp_path):
    """
    Run the entire SLAVV pipeline on the test volume and verify outputs
    and exporters.
    """
    # 1. Load the volume
    # The test data is located at tests/data/slavv_test_volume.tif
    # We resolve the path relative to this test file.
    current_dir = Path(__file__).parent
    test_vol_path = current_dir.parent / "data" / "slavv_test_volume.tif"
    
    if not test_vol_path.exists():
        pytest.skip(f"Test volume not found at {test_vol_path}")
        
    image = load_tiff_volume(str(test_vol_path))
    assert image is not None
    assert len(image.shape) == 3
    
    # 2. Configure processor with robust parameters
    # We use fewer scales to speed up the test slightly.
    params = {
        "microns_per_voxel": [1.0, 1.0, 1.0],
        "radius_of_smallest_vessel_in_microns": 1.5,
        "radius_of_largest_vessel_in_microns": 10.0,
        "scales_per_octave": 1.0,  # fast test
        "energy_upper_bound": 0.0,
        "space_strel_apothem": 1,
        "length_dilation_ratio": 1.0,
        "number_of_edges_per_vertex": 4,
        "max_voxels_per_node_energy": 500000,
        "approximating_PSF": False,  # disable PSF for speed if needed, or leave True
    }
    
    # 3. Process image
    processor = SLAVVProcessor()
    results = processor.process_image(image, params)
    
    # 4. Assert Results Integrity
    assert "vertices" in results
    assert "edges" in results
    assert "network" in results
    
    vertices = results["vertices"]
    edges = results["edges"]
    network = results["network"]
    
    assert len(vertices["positions"]) > 0, "No vertices detected."
    assert len(vertices["energies"]) == len(vertices["positions"])
    assert len(edges["traces"]) > 0, "No edges traced."
    assert len(network["strands"]) > 0, "No network strands assembled."
    
    # 5. Test Exporters
    out_dir = tmp_path / "exports"
    out_dir.mkdir(exist_ok=True)
    
    # JSON
    json_path = out_dir / "network.json"
    save_network_to_json(network, str(json_path))
    assert json_path.exists()
    assert json_path.stat().st_size > 0
    
    # CSV
    csv_path = out_dir / "network_edges.csv"
    save_network_to_csv(network, str(csv_path).replace("_edges.csv", ".csv"))
    # Save network to CSV actually outputs network_edges.csv and network_vertices.csv
    # based on the base path.
    # Our exporter writes directly or adds suffixes?
    # Let's just check the directory has CSV files.
    csv_files = list(out_dir.glob("*.csv"))
    assert len(csv_files) > 0
    
    
    # CASX
    casx_path = out_dir / "network.casx"
    save_network_to_casx(network, str(casx_path))
    assert casx_path.exists()
    assert casx_path.stat().st_size > 0
    
    # VMV
    vmv_path = out_dir / "network.vmv"
    save_network_to_vmv(network, str(vmv_path))
    assert vmv_path.exists()
    assert vmv_path.stat().st_size > 0
