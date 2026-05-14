from scipy.io import loadmat
import numpy as np
from pathlib import Path

# Path to the MATLAB curated vertices
vertices_path = Path("workspace/oracles/180709_E_batch_190910-103039/01_Input/matlab_results/batch_190910-103039_canonical/vectors/curated_vertices_full.mat")

if not vertices_path.exists():
    print(f"Missing file: {vertices_path}")
    exit(1)

data = loadmat(vertices_path, squeeze_me=True, struct_as_record=False)
if 'vertex_space_subscripts' in data:
    pos = data['vertex_space_subscripts'].astype(np.float64) - 1.0
else:
    exit(1)

extra_pairs = [[21, 92]]

for p in extra_pairs:
    v1, v2 = p
    p1 = pos[v1]
    p2 = pos[v2]
    dist = np.linalg.norm(p1 - p2)
    print(f"Pair {p}: p1={p1}, p2={p2}, dist={dist:.4f}")
