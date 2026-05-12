import joblib
import numpy as np

p_edges = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\final_audit_destination\02_Output\python_results\checkpoints\checkpoint_edge_candidates.pkl")

c = p_edges['connections']
t = p_edges['traces']

# Find our specific divergent connection from 1143
for i, conn in enumerate(c):
    if conn[0] == 1143:
        print(f"FOUND PYTHON TRACE STARTING AT 1143 (Index in list: {i})")
        print(f"CONNECTION ENDS AT: {conn[1]}")
        path = t[i]
        print(f"TRACE DATA TYPE: {path.dtype}")
        print(f"TRACE RAW ARRAY:\n{path}")
        break
