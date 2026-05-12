import joblib
import numpy as np

ckpt_path = r"d:\2P_Data\Aaron\slavv2python\workspace\runs\candidate_fix_audit_v2_final_verification\02_Output\python_results\checkpoints\checkpoint_edge_candidates.pkl"
data = joblib.load(ckpt_path)

print(f"TOTAL CANDIDATES GENERATED: {len(data['connections'])}")
