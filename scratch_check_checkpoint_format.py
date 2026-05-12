import joblib
import numpy as np

p_path = r"d:\2P_Data\Aaron\slavv2python\workspace\runs\candidate_fix_audit_v1\02_Output\python_results\checkpoints\checkpoint_vertices.pkl"
o_path = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\vertices.pkl"

p_data = joblib.load(p_path)
o_data = joblib.load(o_path)

print("PYTHON FORMAT KEYS:")
print(list(p_data.keys()))
for k, v in p_data.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: len={len(v) if hasattr(v, '__len__') else 'N/A'}")

print("\nORACLE FORMAT KEYS:")
print(list(o_data.keys()))
for k, v in o_data.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: len={len(v) if hasattr(v, '__len__') else 'N/A'}")
