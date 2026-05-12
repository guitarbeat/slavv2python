import joblib
import numpy as np

# Oracle paths
oracle_path = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\edges.pkl"

# Python paths for current attempt
python_path = r"d:\2P_Data\Aaron\slavv2python\workspace\runs\revert_to_baseline_trial\02_Output\python_results\checkpoints\checkpoint_edge_candidates.pkl"

oracle_data = joblib.load(oracle_path)
python_data = joblib.load(python_path)

oracle_c = oracle_data['connections']
python_c = python_data['connections']

print(f"ORACLE CONNECTION COUNT: {len(oracle_c)}")
print(f"PYTHON CONNECTION COUNT: {len(python_c)}")

def normalize_connections(c):
    norm = []
    for pair in c:
        norm.append(tuple(sorted(list(pair))))
    return set(norm)

oracle_set = normalize_connections(oracle_c)
python_set = normalize_connections(python_c)

overlap = oracle_set.intersection(python_set)
matched_count = len(overlap)

print(f"TOTAL ORACLE SET SIZE (UNIQUE): {len(oracle_set)}")
print(f"TOTAL PYTHON SET SIZE (UNIQUE): {len(python_set)}")
print(f"INTERSECTION (MATCHED): {matched_count}")
print(f"PARITY PERCENTAGE: {matched_count / len(oracle_set) * 100.0:.2f}%")
