from scipy.io import loadmat
import numpy as np
from pathlib import Path

# Path to the MATLAB edges
oracle_root = Path("workspace/oracles/180709_E_batch_190910-103039/01_Input/matlab_results/batch_190910-103039_canonical/vectors")
edges_path = oracle_root / "edges_190910-225419_tie2gfp16 9juyly2018 870nm region a-082-1.mat"

if not edges_path.exists():
    print(f"Missing file: {edges_path}")
    exit(1)

data = loadmat(edges_path, squeeze_me=True, struct_as_record=False)

# In edges.mat, the field is often edges2vertices
edges = data['edges2vertices'] - 1 # 0-indexed

v_target = 21
connected_to = []
for i in range(len(edges)):
    if edges[i, 0] == v_target:
        connected_to.append(int(edges[i, 1]))
    elif edges[i, 1] == v_target:
        connected_to.append(int(edges[i, 0]))

print(f"Vertex {v_target} is connected to in MATLAB (edges.mat): {sorted(connected_to)}")
