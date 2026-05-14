from scipy.io import loadmat
import numpy as np
from pathlib import Path

# Path to the MATLAB vertex curation
vertex_file = "workspace/oracles/180709_E_batch_190910-103039/01_Input/matlab_results/batch_190910-103039_canonical/curations/vertices_190910-172151_tie2gfp16 9juyly2018 870nm region a-082-1.mat"

data = loadmat(vertex_file, squeeze_me=True, struct_as_record=False)
energies = data['vertex_energies']

v_target = 92
neighbors = [20, 21, 35, 86, 257]
print(f"Vertex {v_target} energy: {energies[v_target]}")
for n in neighbors:
    print(f"Neighbor {n} energy: {energies[n]}")
