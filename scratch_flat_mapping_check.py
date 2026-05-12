import joblib
import numpy as np
from slavv_python.core.global_watershed import _coord_to_matlab_linear_index

e_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl")
scale_indices = e_data.get('scale_indices')

shape = scale_indices.shape
print(f"VOLUME SHAPE: {shape}")

# Center pixel for Node 92
z, y, x = 54, 375, 335
coord = np.array([z, y, x], dtype=np.int32)

# The function used by the runner
linear_idx = _coord_to_matlab_linear_index(coord, shape)
print(f"\nCOMPUTED LINEAR INDEX (1-BASED): {linear_idx}")

# RAVEL order="F" matches MATLAB memory layout
flat_scales = scale_indices.ravel(order='F')

# Python 0-based lookup
extracted_val = flat_scales[linear_idx - 1]
print(f"EXTRACTED VALUE FROM FLAT VECTOR: {extracted_val}")

direct_val = scale_indices[z, y, x]
print(f"DIRECT VALUE FROM 3D VOLUME    : {direct_val}")

# IF THEY DIFFER, THE ENTIRE FLAT MAPPING IS BROKEN!
