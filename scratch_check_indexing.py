import joblib
import numpy as np

# Load our final run's produced candidates
python_ckpt = r"d:\2P_Data\Aaron\slavv2python\workspace\runs\candidate_fix_audit_v2_final_verification\02_Output\python_results\checkpoints\checkpoint_edge_candidates.pkl"
py_data = joblib.load(python_ckpt)
py_conn = np.asarray(py_data['connections'])

# Load oracle's preserved edges
oracle_path = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\edges.pkl"
ora_data = joblib.load(oracle_path)
ora_conn = np.asarray(ora_data['connections'])

print(f"Python Index Min: {py_conn.min()}, Max: {py_conn.max()}")
print(f"Oracle Index Min: {ora_conn.min()}, Max: {ora_conn.max()}")
