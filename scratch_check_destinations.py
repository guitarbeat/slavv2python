import joblib
import numpy as np

v_path = r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl"
v_data = joblib.load(v_path)
v_pos = v_data['positions']

print(f"START VERTEX 1143: {v_pos[1143]}")
print(f"PYTHON DEST VERTEX 1033: {v_pos[1033]}")
print(f"ORACLE DEST VERTEX 1067: {v_pos[1067]}")

print("\nWait, is [63, 272, 42] in the vertex list anywhere?")
matches = np.where(np.all(np.abs(v_pos - np.array([63.0, 272.0, 42.0])) < 1.0, axis=1))[0]
print(f"Matches found at indices: {matches}")
