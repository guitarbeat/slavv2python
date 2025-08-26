import pathlib
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit' / 'src'))

from io_utils import load_network_from_casx, load_network_from_vmv


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

