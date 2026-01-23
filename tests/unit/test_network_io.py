"""Consolidated tests for network I/O (CSV, JSON, CASX, VMV)."""
import json
import numpy as np
import pytest
from pathlib import Path

from src.slavv.io_utils import (
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
        """Test loading network from CASX text format."""
        # Expected vertices (y, x, z): [[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]]
        # Written as (x, -y, -z):
        # v0: 1.0, -2.0, -3.0
        # v1: 4.0, -5.0, -6.0

        casx_content = (
            "//Header\n"
            "//point coordinates;   nPoints=2\n"
            "\t1.000000000E+00\t-2.000000000E+00\t-3.000000000E+00\n"
            "\t4.000000000E+00\t-5.000000000E+00\t-6.000000000E+00\n"
            "//end point coordinates\n"
            "\n"
            "//arc connectivity matrix;   nArcs=1\n"
            "1\t2\t\n"
            "//end arc connectivity matrix\n"
            "\n"
            "//diameter: vector on arc;   nArcs=1\n"
            "11.0\n"
            "//end diameter\n"
        )

        casx_path = tmp_path / "network.casx"
        casx_path.write_text(casx_content)

        network = load_network_from_casx(casx_path)

        expected_vertices = np.array([[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]], dtype=float)
        assert np.allclose(network.vertices, expected_vertices)
        assert np.array_equal(network.edges, np.array([[0, 1]], dtype=int))

        # Radii are estimated from arc diameters.
        # Diameter 11.0 => Radius 5.5 for arc.
        # Both vertices connected to this arc get radius 5.5.
        assert np.allclose(network.radii, np.array([5.5, 5.5], dtype=float))

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

    def test_casx_roundtrip(self, tmp_path: Path) -> None:
        """Test full roundtrip export -> import for CASX."""
        from src.slavv.visualization import NetworkVisualizer

        viz = NetworkVisualizer()

        # Setup data
        # Vertices (y, x, z) in voxels
        vertices = {
            'positions': np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
            'energies': np.array([0.1, 0.2]),
            'radii': np.array([1.5, 2.5]),
            'scales': np.array([1, 1])
        }
        # Edges (trace points in voxels)
        edges = {
            'connections': [[0, 1]],
            'traces': [np.array([[10.0, 20.0, 30.0], [25.0, 35.0, 45.0], [40.0, 50.0, 60.0]])],
            'energies': [0.15]
        }
        network = {
            'strands': [[0, 1]],
            'bifurcations': []
        }
        # Voxel size
        microns_per_voxel = [0.5, 0.5, 2.0]
        parameters = {
            'microns_per_voxel': microns_per_voxel
        }

        out_path = str(tmp_path / "roundtrip.casx")
        viz._export_casx(vertices, edges, network, parameters, out_path)

        # Load back
        loaded = load_network_from_casx(out_path)

        # Verify vertices
        # Original (y, x, z) in voxels -> * microns -> (y_um, x_um, z_um)
        # v0: (5.0, 10.0, 60.0)
        # v1: (20.0, 25.0, 120.0)
        # Intermediate: (12.5, 17.5, 90.0)

        # Loader returns ALL points (terminals + intermediates)
        # We expect 3 points.
        assert len(loaded.vertices) == 3

        # Implementation adds all terminal vertices first, then intermediate points.
        # Terminals: v0, v1. (Sorted by index).
        # v0 -> index 0.
        # v1 -> index 1.
        # Intermediate -> index 2.

        # Point 0: v0
        expected_v0 = np.array([5.0, 10.0, 60.0])
        assert np.allclose(loaded.vertices[0], expected_v0)

        # Point 1: v1
        expected_v1 = np.array([20.0, 25.0, 120.0])
        assert np.allclose(loaded.vertices[1], expected_v1)

        # Point 2: Intermediate
        expected_mid = np.array([12.5, 17.5, 90.0])
        assert np.allclose(loaded.vertices[2], expected_mid)

        # Connectivity: 0-2, 2-1 (based on indices: v0->mid, mid->v1)
        # 0->2
        # 2->1
        # Edges might be loaded in order of arcs in file.
        # File arcs: v0->mid, mid->v1.
        # So edges[0]: [0, 2]
        # edges[1]: [2, 1]
        assert len(loaded.edges) == 2
        assert np.array_equal(loaded.edges[0], [0, 2])
        assert np.array_equal(loaded.edges[1], [2, 1])

        # Radii (estimated)
        # Trace radii interpolation.
        # r0 = 1.5, r1 = 2.5.
        # mid point dist?
        # dists:
        # v0->mid: dy=7.5, dx=7.5, dz=30. norm = sqrt(7.5^2+7.5^2+30^2)
        # mid->v1: same.
        # So mid is exactly half way.
        # r_mid = (1.5 + 2.5)/2 = 2.0.
        # Arc 1 (v0-mid): avg radius (1.5+2.0)/2 = 1.75.
        # Arc 2 (mid-v1): avg radius (2.0+2.5)/2 = 2.25.

        # Estimated vertex radii:
        # v0 (idx 0): connected to arc 1. Radius = 1.75.
        # v1 (idx 1): connected to arc 2. Radius = 2.25.
        # mid (idx 2): connected to arc 1 (1.75) and arc 2 (2.25). Avg = 2.0.

        assert np.allclose(loaded.radii, np.array([1.75, 2.25, 2.0]), atol=0.01)

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
