import numpy as np
from slavv.visualization.volume_rasterization import paint_vertices_to_volume, paint_edges_to_volume

def test_paint_vertices_to_volume():
    # Setup
    shape = (10, 10, 10)
    vertices = {
        'positions': np.array([[5.0, 5.0, 5.0]]),
        'radii_pixels': np.array([2.0]),
        'energies': np.array([0.8])
    }

    # Test filling with energy
    vol = paint_vertices_to_volume(vertices, shape, fill_value='energy')
    assert vol[5, 5, 5] == 0.8
    assert vol[0, 0, 0] == 0.0

    # Check radius approximately
    # Center is (5,5,5). Radius 2.
    # (7,5,5) dist is 2. (7-5)^2 = 4. <= 4. Should be painted.
    assert vol[7, 5, 5] == 0.8
    # (8,5,5) dist is 3. > 2. Should be 0.
    assert vol[8, 5, 5] == 0.0

def test_paint_edges_to_volume():
    # Setup
    shape = (10, 10, 10)
    # Edge from (2,2,2) to (2,2,8) along Z axis
    trace = np.array([[2.0, 2.0, z] for z in range(2, 9)])
    edges = {
        'traces': [trace],
        'energies': [0.5],
        'connections': [[0, 1]]
    }
    vertices = {
        'radii_pixels': np.array([1.0, 1.0])
    }

    vol = paint_edges_to_volume(edges, vertices, shape, fill_value='energy')

    # Check points on trace
    assert vol[2, 2, 5] == 0.5

    # Check radius around trace (radius 1)
    # (2, 3, 5) dist 1.
    assert vol[2, 3, 5] == 0.5
    # (2, 4, 5) dist 2.
    assert vol[2, 4, 5] == 0.0

def test_paint_vertices_fill_value():
    shape = (5, 5, 5)
    vertices = {
        'positions': np.array([[2.0, 2.0, 2.0]]),
        'radii_pixels': np.array([1.0])
    }

    # Constant fill
    vol = paint_vertices_to_volume(vertices, shape, fill_value=0.5)
    assert vol[2, 2, 2] == 0.5

    # Radius fill
    vol = paint_vertices_to_volume(vertices, shape, fill_value='radius')
    assert vol[2, 2, 2] == 1.0

def test_empty_input():
    shape = (5, 5, 5)
    vertices = {'positions': np.array([])}
    vol = paint_vertices_to_volume(vertices, shape)
    assert np.all(vol == 0)

    edges = {'traces': []}
    vol = paint_edges_to_volume(edges, vertices, shape)
    assert np.all(vol == 0)
