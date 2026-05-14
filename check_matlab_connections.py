from scipy.io import loadmat
import numpy as np
from pathlib import Path

# Path to the MATLAB curated vertices and edges
oracle_root = Path("workspace/oracles/180709_E_batch_190910-103039/01_Input/matlab_results/batch_190910-103039_canonical/vectors")
vertices_path = oracle_root / "curated_vertices_full.mat"
edges_path = oracle_root / "curated_vertices_full.mat" # Wait, curated_vertices_full.mat might have edges too

if not vertices_path.exists():
    print(f"Missing file: {vertices_path}")
    exit(1)

data = loadmat(vertices_path, squeeze_me=True, struct_as_record=False)

# Check for edges2vertices
if hasattr(data, 'edges2vertices'):
    # loadmat with struct_as_record=False returns a dict-like object but attributes might be accessed via dot
    # Actually, if it's a dict, use keys.
    pass

# Let's just use dict access
edges = data['edges2vertices'] - 1 # 0-indexed

v_target = 25
connected_to = []
for i in range(len(edges)):
    if edges[i, 0] == v_target:
        connected_to.append(int(edges[i, 1]))
    elif edges[i, 1] == v_target:
        connected_to.append(int(edges[i, 0]))

print(f"Vertex {v_target} is connected to in MATLAB: {sorted(connected_to)}")
