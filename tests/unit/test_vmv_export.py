
import os
import numpy as np
from slavv.visualization import NetworkVisualizer

def test_vmv_export():
    viz = NetworkVisualizer()

    # Mock data
    # Coordinates in voxels: (y, x, z)
    # We use microns_per_voxel = 1.0, so coords are same in microns.
    # Output expected: x, y, z
    vertices = {
        'positions': np.array([
            [10, 20, 30], # Node 0: y=10, x=20, z=30
            [15, 25, 35], # Node 1: y=15, x=25, z=35
            [20, 30, 40]  # Node 2: y=20, x=30, z=40
        ]),
        'energies': np.array([0.5, 0.6, 0.7]),
        'radii_microns': np.array([1.0, 1.2, 1.4]),
        'scales': np.array([1, 1, 1])
    }

    # Trace points include endpoints.
    # Edge 0-1:
    edges = {
        'traces': [
            np.array([[10, 20, 30], [12.5, 22.5, 32.5], [15, 25, 35]]), # y,x,z. Midpoint included.
            np.array([[15, 25, 35], [20, 30, 40]])
        ],
        'connections': [
            [0, 1],
            [1, 2]
        ],
        'energies': [0.55, 0.65]
    }

    network = {
        'strands': [
            [0, 1, 2] # One strand connecting 0->1->2
        ],
        'bifurcations': [],
        'vertex_degrees': []
    }

    parameters = {
        'microns_per_voxel': [1.0, 1.0, 1.0]
    }

    output_path = "test_export.vmv"
    if os.path.exists(output_path):
        os.remove(output_path)

    # Run export
    viz._export_vmv(vertices, edges, network, parameters, output_path)

    # Read and verify content
    with open(output_path, 'r') as f:
        content = f.read()

    lines = content.strip().split('\n')

    # Check Header
    assert lines[0] == "$PARAM_BEGIN"
    params = {}
    idx = 1
    while lines[idx] != "$PARAM_END":
        parts = lines[idx].split('\t')
        params[parts[0]] = int(parts[1])
        idx += 1

    assert params['NUM_VERTS'] >= 3
    assert params['NUM_STRANDS'] == 1
    assert params['NUM_ATTRIB_PER_VERT'] == 4

    idx += 1 # skip $PARAM_END
    while idx < len(lines) and not lines[idx].strip():
        idx += 1 # skip empty lines

    # Check Vertices
    assert lines[idx] == "$VERT_LIST_BEGIN"
    idx += 1

    points = []
    while lines[idx] != "$VERT_LIST_END":
        parts = lines[idx].split('\t')
        # Index, x, y, z, r
        int(parts[0])
        x, y, z, r = map(float, parts[1:])
        points.append((x, y, z, r))
        idx += 1

    assert len(points) == params['NUM_VERTS']
    # Check coordinates swap (y,x,z) -> (x, -y, -z) matching MATLAB spec
    # Node 0 was (10, 20, 30) -> should be (20, -10, -30)
    # But wait, trace had midpoint (12.5, 22.5, 32.5) -> (22.5, -12.5, -32.5)
    # We should have at least:
    # (20, -10, -30)
    # (22.5, -12.5, -32.5)
    # (25, -15, -35)
    # (30, -20, -40)

    # Let's verify presence
    found_start = False
    for p in points:
        if np.allclose(p[:3], [20.0, -10.0, -30.0], atol=1e-5):
            found_start = True
    assert found_start, "Start point (20, -10, -30) not found in VMV output"

    idx += 1 # skip $VERT_LIST_END
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    # Check Strands
    assert lines[idx] == "$STRANDS_LIST_BEGIN"
    idx += 1

    strands = []
    while idx < len(lines) and lines[idx] != "$STRANDS_LIST_END":
        parts = lines[idx].split('\t')
        int(parts[0])
        pt_indices = list(map(int, parts[1:]))
        strands.append(pt_indices)
        idx += 1

    assert len(strands) == 1
    # Check strand continuity
    strand_pts = strands[0]
    # Should have 4 points (3 from edge1 + 2 from edge2 - 1 shared) = 4
    # edge1 has 3 points. edge2 has 2 points.
    # Total unique points should be 4.
    assert len(strand_pts) == 4, f"Expected 4 points in strand, got {len(strand_pts)}"

    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)
