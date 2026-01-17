import json
import pathlib
import sys
import numpy as np
from pathlib import Path

from src.slavv.io_utils import (
    load_network_from_casx,
    load_network_from_vmv,
    load_network_from_csv,
    load_network_from_json,
)


def test_load_network_from_casx(tmp_path: Path) -> None:
    xml = (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<CasX><Network><Vertices>"
        "<Vertex id='0' x='1.0' y='2.0' z='3.0' radius='4.0'/>"
        "<Vertex id='1' x='4.0' y='5.0' z='6.0' radius='7.0'/>"
        "</Vertices><Edges>"
        "<Edge id='0' start='0' end='1'/>"
        "</Edges></Network></CasX>"
    )
    casx_path = tmp_path / "network.casx"
    casx_path.write_text(xml)

    network = load_network_from_casx(casx_path)

    expected_vertices = np.array([[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]], dtype=float)
    expected_edges = np.array([[0, 1]], dtype=int)
    expected_radii = np.array([4.0, 7.0], dtype=float)

    assert np.allclose(network.vertices, expected_vertices)
    assert np.array_equal(network.edges, expected_edges)
    assert np.allclose(network.radii, expected_radii)


def test_load_network_from_csv(tmp_path: Path) -> None:
    prefix = tmp_path / "net"
    vertex_csv = prefix.with_name("net_vertices.csv")
    edge_csv = prefix.with_name("net_edges.csv")
    vertex_csv.write_text(
        "vertex_id,y_position,x_position,z_position,energy,radius_microns,scale\n"
        "0,1.0,2.0,3.0,0.1,4.0,1.0\n"
        "1,4.0,5.0,6.0,0.2,7.0,1.0\n"
    )
    edge_csv.write_text(
        "edge_id,start_vertex,end_vertex,length,n_points\n0,0,1,1.0,2\n"
    )

    network = load_network_from_csv(prefix)

    expected_vertices = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    expected_edges = np.array([[0, 1]], dtype=int)
    expected_radii = np.array([4.0, 7.0], dtype=float)

    assert np.allclose(network.vertices, expected_vertices)
    assert np.array_equal(network.edges, expected_edges)
    assert np.allclose(network.radii, expected_radii)


def test_load_network_from_json(tmp_path: Path) -> None:
    data = {
        "vertices": {
            "positions": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "radii_microns": [4.0, 7.0],
        },
        "edges": {"connections": [[0, 1]]},
    }
    json_path = tmp_path / "net.json"
    json_path.write_text(json.dumps(data))

    network = load_network_from_json(json_path)

    expected_vertices = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    expected_edges = np.array([[0, 1]], dtype=int)
    expected_radii = np.array([4.0, 7.0], dtype=float)

    assert np.allclose(network.vertices, expected_vertices)
    assert np.array_equal(network.edges, expected_edges)
    assert np.allclose(network.radii, expected_radii)


def test_load_network_from_vmv(tmp_path: Path) -> None:
    text = (
        "# VMV Format Export\n"
        "[VERTICES]\n"
        "0 1.0 2.0 3.0 4.0 0.5\n"
        "1 5.0 6.0 7.0 8.0 0.6\n\n"
        "[EDGES]\n"
        "0 0 1\n"
    )
    vmv_path = tmp_path / "network.vmv"
    vmv_path.write_text(text)

    network = load_network_from_vmv(vmv_path)

    expected_vertices = np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]], dtype=float)
    expected_edges = np.array([[0, 1]], dtype=int)
    expected_radii = np.array([4.0, 8.0], dtype=float)

    assert np.allclose(network.vertices, expected_vertices)
    assert np.array_equal(network.edges, expected_edges)
    assert np.allclose(network.radii, expected_radii)

