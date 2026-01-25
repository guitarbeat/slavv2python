import os
import numpy as np
import pytest
import xml.etree.ElementTree as ET
from src.slavv.visualization import NetworkVisualizer

def test_casx_export_full(tmp_path):
    viz = NetworkVisualizer()

    # Mock data
    vertices = {
        'positions': np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]), # Y, X, Z
        'radii_microns': np.array([1.5, 2.5, 3.5]),
        'energies': np.array([0.1, 0.2, 0.3])
    }
    edges = {
        'connections': [[0, 1], [1, 2]],
        'traces': [
            [[10.0, 20.0, 30.0], [25.0, 35.0, 45.0], [40.0, 50.0, 60.0]],
            [[40.0, 50.0, 60.0], [55.0, 65.0, 75.0], [70.0, 80.0, 90.0]]
        ],
        'energies': [0.15, 0.25]
    }
    network = {
        'strands': [[0, 1, 2]],
        'bifurcations': [],
        'vertex_degrees': [1, 2, 1]
    }
    parameters = {
        'microns_per_voxel': [1.0, 1.0, 1.0]
    }

    output_path = tmp_path / "test_output.casx"
    viz._export_casx(vertices, edges, network, parameters, str(output_path))

    assert output_path.exists()

    tree = ET.parse(output_path)
    root = tree.getroot()

    # Check Root
    assert root.tag == "CasX"

    # Check Parameters
    params = root.find("Parameters")
    assert params is not None
    mpv = params.find(".//Parameter[@name='microns_per_voxel']")
    assert mpv is not None
    assert mpv.attrib['value'] == "1.0 1.0 1.0"

    # Check Network
    net = root.find("Network")
    assert net is not None

    # Check Vertices
    verts_elem = net.find("Vertices")
    assert len(verts_elem.findall("Vertex")) == 3
    v0 = verts_elem.find("Vertex[@id='0']")
    # Check coordinate swap: X should be pos[1] (20.0), Y should be pos[0] (10.0)
    assert float(v0.attrib['x']) == 20.0
    assert float(v0.attrib['y']) == 10.0

    # Check Edges
    edges_elem = net.find("Edges")
    assert len(edges_elem.findall("Edge")) == 2

    # Check Strands
    strands_elem = net.find("Strands")
    assert strands_elem is not None
    strand_list = strands_elem.findall("Strand")
    assert len(strand_list) == 1
    s0 = strand_list[0]
    assert s0.attrib['id'] == "0"
    assert s0.text.strip() == "0 1 2"
