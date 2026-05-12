import joblib
import numpy as np

def load_edges(path):
    data = joblib.load(path)
    if 'connections' in data:
        return np.asarray(data['connections'])
    return np.zeros((0,2))

# Load our final run's produced candidates
python_ckpt = r"d:\2P_Data\Aaron\slavv2python\workspace\runs\candidate_fix_audit_v2_final_verification\02_Output\python_results\checkpoints\checkpoint_edge_candidates.pkl"
py_data = joblib.load(python_ckpt)
py_conn = np.asarray(py_data['connections'])

# Load oracle's preserved edges
oracle_path = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\edges.pkl"
ora_data = joblib.load(oracle_path)
ora_conn = np.asarray(ora_data['connections'])

target_v = 1350
print(f"CHECKING FOR CONNECTIONS FROM VERTEX {target_v}...")

py_matches = py_conn[(py_conn[:, 0] == target_v) | (py_conn[:, 1] == target_v)]
print(f"PYTHON CONNECTIONS FOR {target_v}: {py_matches.tolist()}")

ora_matches = ora_conn[(ora_conn[:, 0] == target_v) | (ora_conn[:, 1] == target_v)]
print(f"ORACLE CONNECTIONS FOR {target_v}: {ora_matches.tolist()}")

print("\nNOW COUNTING OVERALL MATCHING STATISTICS...")
print(f"Python Total Candidates: {len(py_conn)}")
print(f"Oracle Total Candidates: {len(ora_conn)}")

# Normalize pairs to compare
py_norm = set(tuple(sorted(x)) for x in py_conn.tolist())
ora_norm = set(tuple(sorted(x)) for x in ora_conn.tolist())

overlap = py_norm.intersection(ora_norm)
missing = ora_norm - py_norm
extra = py_norm - ora_norm

print(f"Matches: {len(overlap)}")
print(f"Oracle candidates NOT found by Python: {len(missing)}")
print(f"Python candidates NOT found in Oracle: {len(extra)}")
