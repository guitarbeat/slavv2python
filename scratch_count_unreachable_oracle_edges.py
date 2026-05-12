import joblib
import numpy as np

edges_path = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\edges.pkl"
edges = joblib.load(edges_path)
c = edges['connections']

max_valid = 1312 # 0-based indexing, max vertex is length - 1 (1313 - 1)
total_oracle_edges = len(c)

# Check how many oracle edges refer to AT LEAST ONE vertex that DOES NOT EXIST IN THE INPUT VERTEX LIST
unreachable_mask = (c[:, 0] > max_valid) | (c[:, 1] > max_valid)
unreachable_count = np.sum(unreachable_mask)

print(f"TOTAL ORACLE EDGES: {total_oracle_edges}")
print(f"ORACLE EDGES REFERRING TO MISSING VERTICES (>1312): {unreachable_count}")
print(f"ORACLE EDGES REMAINING (FULLY WITHIN BOUNDS 0-1312): {total_oracle_edges - unreachable_count}")
