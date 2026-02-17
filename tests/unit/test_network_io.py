"""Consolidated tests for network I/O (CSV, JSON, CASX, VMV)."""
import json
import numpy as np
from pathlib import Path

from slavv.io import (
    Network,
    load_network_from_casx,
    load_network_from_vmv,
    load_network_from_csv,
    load_network_from_json,
    save_network_to_csv,
    save_network_to_json,
)


class TestNetworkRoundtrip:
    """Test save/load roundtrip for network formats."""
    
    def test_csv_roundtrip(self, tmp_path: Path) -> None:
        """Test CSV save and load preserves data."""
        network = Network(
            vertices=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
            edges=np.array([[0, 1]], dtype=int),
            radii=np.array([4.0, 7.0], dtype=float),
        )

        v_path, e_path = save_network_to_csv(network, tmp_path / 'net')
        assert v_path.exists() and e_path.exists()

        loaded = load_network_from_csv(tmp_path / 'net')
        assert np.allclose(loaded.vertices, network.vertices)
        assert np.array_equal(loaded.edges, network.edges)
        assert np.allclose(loaded.radii, network.radii)

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        """Test JSON save and load preserves data."""
        network = Network(
            vertices=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
            edges=np.array([[0, 1]], dtype=int),
            radii=np.array([4.0, 7.0], dtype=float),
        )

        path = save_network_to_json(network, tmp_path / 'net.json')
        assert Path(path).exists()

        loaded = load_network_from_json(path)
        assert np.allclose(loaded.vertices, network.vertices)
        assert np.array_equal(loaded.edges, network.edges)
        assert np.allclose(loaded.radii, network.radii)


class TestNetworkImport:
    """Test loading networks from various formats."""
    
    def test_load_from_casx(self, tmp_path: Path) -> None:
        """Test loading network from CASX XML format."""
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
        assert np.allclose(network.vertices, expected_vertices)
        assert np.array_equal(network.edges, np.array([[0, 1]], dtype=int))
        assert np.allclose(network.radii, np.array([4.0, 7.0], dtype=float))

    def test_load_from_vmv(self, tmp_path: Path) -> None:
        """Test loading network from VMV text format."""
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
        assert np.allclose(network.vertices, expected_vertices)
        assert np.array_equal(network.edges, np.array([[0, 1]], dtype=int))
        assert np.allclose(network.radii, np.array([4.0, 8.0], dtype=float))

    def test_load_from_json_file(self, tmp_path: Path) -> None:
        """Test loading network from JSON format."""
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
        assert np.allclose(network.vertices, expected_vertices)
        assert np.array_equal(network.edges, np.array([[0, 1]], dtype=int))
        assert np.allclose(network.radii, np.array([4.0, 7.0], dtype=float))

    def test_load_from_csv_files(self, tmp_path: Path) -> None:
        """Test loading network from CSV format."""
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
        assert np.allclose(network.vertices, expected_vertices)
        assert np.array_equal(network.edges, np.array([[0, 1]], dtype=int))
        assert np.allclose(network.radii, np.array([4.0, 7.0], dtype=float))
