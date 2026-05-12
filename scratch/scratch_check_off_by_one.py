import joblib
import numpy as np

v_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")
v_pos = v_data['positions']

# Let's print offsets around 91-92
print(f"V_POS[91]: {v_pos[91]}")
print(f"V_POS[92]: {v_pos[92]}")

from slavv_python.core.global_watershed import _coord_to_matlab_linear_index
shape = (64, 512, 512)

c91 = np.rint(v_pos[91]).astype(int)
c92 = np.rint(v_pos[92]).astype(int)

print(f"LINEAR FOR OFFSET 91: {_coord_to_matlab_linear_index(c91, shape)}")
print(f"LINEAR FOR OFFSET 92: {_coord_to_matlab_linear_index(c92, shape)}")
