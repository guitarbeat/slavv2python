import joblib
import numpy as np

p_path = r"d:\2P_Data\Aaron\slavv2python\workspace\runs\candidate_fix_audit_v1\02_Output\python_results\checkpoints\checkpoint_vertices.pkl"
p_data = joblib.load(p_path)

print(f"Sample radii_microns: {p_data['radii_microns'][:5]}")
print(f"Sample radii: {p_data['radii'][:5]}")
print(f"Unique scales in python: {np.unique(p_data['scales'])}")

o_path = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\vertices.pkl"
o_data = joblib.load(o_path)
print(f"Unique scales in oracle: {np.unique(o_data['scales'])}")
