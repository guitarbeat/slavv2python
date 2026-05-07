import xml.etree.ElementTree as ET

import numpy as np
import pytest

from slavv_python.visualization.network_plots import NetworkVisualizer


def test_casx_export(tmp_path):
    output_path = tmp_path / "test_export.casx"
    viz = NetworkVisualizer()

    # Mock data
    vertices = {
        "positions": np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=float),  # y, x, z
        "radii": np.array([1.5, 2.5, 3.5], dtype=float),
        "energies": np.array([0.8, 0.9, 0.7], dtype=float),
        "scales": np.array([1.0, 1.2, 1.4], dtype=float),
    }

    edges = {
        "connections": [[0, 1], [1, 2]],
        # traces are not exported to CASX but usually present
        "traces": [np.array([[10, 20, 30], [40, 50, 60]]), np.array([[40, 50, 60], [70, 80, 90]])],
    }

    network = {"strands": [[0, 1, 2]], "bifurcations": [1], "vertex_degrees": [1, 2, 1]}

    parameters = {"microns_per_voxel": [0.5, 0.5, 1.0], "threshold": 0.5, "algorithm": "SLAVV_v2"}

    processing_results = {
        "vertices": vertices,
        "edges": edges,
        "network": network,
        "parameters": parameters,
    }

    # Run export
    viz.export_network_data(processing_results, str(output_path), format="casx")

    # Verify content
    tree = ET.parse(output_path)
    root = tree.getroot()

    # Check Parameters
    params = {p.attrib["name"]: p.attrib["value"] for p in root.findall(".//Parameter")}
    assert params["microns_per_voxel"] == "0.5 0.5 1.0"
    assert params["threshold"] == "0.5"
    assert params["algorithm"] == "SLAVV_v2"

    # Check Vertices
    verts = root.findall(".//Vertex")
    assert len(verts) == 3

    # Check first vertex attributes
    # Note: x=pos[1], y=pos[0]
    v0 = verts[0].attrib
    assert v0["id"] == "0"
    assert float(v0["x"]) == pytest.approx(20.0, 0.001)
    assert float(v0["y"]) == pytest.approx(10.0, 0.001)
    assert float(v0["z"]) == pytest.approx(30.0, 0.001)
    assert float(v0["radius"]) == pytest.approx(1.5, 0.001)
    assert float(v0["energy"]) == pytest.approx(0.8, 0.001)
    assert float(v0["scale"]) == pytest.approx(1.0, 0.001)

    # Check second vertex attributes
    v1 = verts[1].attrib
    assert float(v1["energy"]) == pytest.approx(0.9, 0.001)

    # Check Edges
    edges_xml = root.findall(".//Edge")
    assert len(edges_xml) == 2
    e0 = edges_xml[0].attrib
    assert e0["id"] == "0"
    assert e0["start"] == "0"
    assert e0["end"] == "1"

    # Check Strands
    strands = root.findall(".//Strand")
    assert len(strands) == 1
    assert strands[0].text == "0 1 2"

    # Check Bifurcations
    bifs = root.findall(".//Bifurcation")
    assert len(bifs) == 1
    assert bifs[0].attrib["vertex_id"] == "1"


def test_casx_export_defaults_missing_radii_to_zero(tmp_path):
    output_path = tmp_path / "missing_radii.casx"
    viz = NetworkVisualizer()

    vertices = {
        "positions": np.array([[10, 20, 30], [40, 50, 60]], dtype=float),
    }
    edges = {"connections": [[0, 1]], "traces": []}
    network = {"strands": [[0, 1]]}

    viz.export_network_data(
        {
            "vertices": vertices,
            "edges": edges,
            "network": network,
            "parameters": {},
        },
        str(output_path),
        format="casx",
    )

    tree = ET.parse(output_path)
    verts = tree.getroot().findall(".//Vertex")
    assert len(verts) == 2
    assert float(verts[0].attrib["radius"]) == pytest.approx(0.0, 0.001)
    assert float(verts[1].attrib["radius"]) == pytest.approx(0.0, 0.001)
