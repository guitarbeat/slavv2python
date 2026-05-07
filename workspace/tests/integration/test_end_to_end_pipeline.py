"""End-to-end integration test for the full SLAVV pipeline.

Loads the real test volume, runs every stage (energy -> vertices -> edges -> network),
and verifies that all export formats produce valid output files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from slavv_python.core import SLAVVProcessor
from slavv_python.io import (
    Network,
    load_tiff_volume,
    save_network_to_casx,
    save_network_to_csv,
    save_network_to_json,
    save_network_to_vmv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TEST_VOLUME = DATA_DIR / "slavv_test_volume.tif"


def _results_to_network(results: dict) -> Network:
    """Convert the raw pipeline results dict into a Network dataclass for exporters."""
    verts = results.get("vertices", {})
    edges = results.get("edges", {})

    positions = np.asarray(verts.get("positions", []), dtype=float)
    radii = np.asarray(verts.get("radii_microns", []), dtype=float)

    # Build a simple edge list from connections (N x 2)
    connections = edges.get("connections", [])
    if len(connections) == 0:
        edge_arr = np.empty((0, 2), dtype=int)
    else:
        edge_arr = np.atleast_2d(np.asarray(connections, dtype=int))

    return Network(
        vertices=positions if positions.size > 0 else np.empty((0, 3)),
        edges=edge_arr,
        radii=radii if radii.size > 0 else None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TEST_VOLUME.exists(), reason="Test volume not available")
def test_end_to_end_pipeline(tmp_path):
    """Run the full SLAVV pipeline and validate all outputs + exporters."""

    # 1. Load volume
    image = load_tiff_volume(str(TEST_VOLUME))
    assert image is not None
    assert image.ndim == 3

    # Use a small central crop to keep the test fast while still exercising
    # every pipeline stage on real data.
    crop = image[50:120, 100:300, 100:300]  # 70 x 200 x 200

    # 2. Pipeline parameters tuned for speed
    params = {
        "microns_per_voxel": [1.0, 1.0, 1.0],
        "radius_of_smallest_vessel_in_microns": 1.5,
        "radius_of_largest_vessel_in_microns": 5.0,  # small range -> fewer scales
        "scales_per_octave": 1.0,
        "energy_upper_bound": 0.0,
        "space_strel_apothem": 1,
        "length_dilation_ratio": 1.0,
        "number_of_edges_per_vertex": 4,
        # Set large enough so no chunking occurs (crop is ~2.8M voxels)
        "max_voxels_per_node_energy": 5_000_000,
        "approximating_PSF": False,
    }

    # 3. Process
    processor = SLAVVProcessor()
    results = processor.process_image(crop, params)

    # 4. Validate result structure
    assert "vertices" in results
    assert "edges" in results
    assert "network" in results

    vertices = results["vertices"]
    edges = results["edges"]
    network = results["network"]

    assert len(vertices["energies"]) == len(vertices["positions"])
    assert len(edges["connections"]) == len(edges["traces"])
    assert "vertex_degrees" in network
    assert network["vertex_degrees"].shape[0] == len(vertices["positions"])

    # 5. Test exporters - convert dict -> Network dataclass first
    net_obj = _results_to_network(results)
    out_dir = tmp_path / "exports"
    out_dir.mkdir()

    # JSON
    json_path = out_dir / "network.json"
    save_network_to_json(results, str(json_path))
    assert json_path.exists()
    assert json_path.stat().st_size > 0

    # CSV
    csv_base = str(out_dir / "network")
    save_network_to_csv(net_obj, csv_base)
    csv_files = list(out_dir.glob("*.csv"))
    assert csv_files

    # CASX
    casx_path = out_dir / "network.casx"
    save_network_to_casx(net_obj, str(casx_path))
    assert casx_path.exists()
    assert casx_path.stat().st_size > 0

    # VMV
    vmv_path = out_dir / "network.vmv"
    save_network_to_vmv(net_obj, str(vmv_path))
    assert vmv_path.exists()
    assert vmv_path.stat().st_size > 0
