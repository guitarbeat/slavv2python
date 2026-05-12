import joblib
import numpy as np

o_v = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\vertices.pkl")['positions']
p_v = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")['positions']

print(f"ORACLE VERTEX 1143: {o_v[1143]}")
print(f"PYTHON VERTEX 1143: {p_v[1143]}")

print("\nFIRST 5 ORACLE VERTICES:")
for i in range(5): print(f"{i}: {o_v[i]}")

print("\nFIRST 5 PYTHON VERTICES:")
for i in range(5): print(f"{i}: {p_v[i]}")

print(f"\nMATCH: {np.array_equal(o_v, p_v)}")
