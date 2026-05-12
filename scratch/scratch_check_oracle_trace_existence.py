import joblib
import numpy as np

oracle_edges_path = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\edges.pkl"
o_edges = joblib.load(oracle_edges_path)

print("ORACLE EDGES KEYS:")
print(list(o_edges.keys()))
if "traces" in o_edges:
    print(f"Found 'traces' key! Type: {type(o_edges['traces'])}")
    if isinstance(o_edges['traces'], list) and len(o_edges['traces']) > 0:
        print(f"Sample trace shape: {np.asarray(o_edges['traces'][0]).shape}")
else:
    print("NO 'traces' key found in oracle edges. Checking other keys for path data.")
