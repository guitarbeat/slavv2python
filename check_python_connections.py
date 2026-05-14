import joblib
from pathlib import Path
import numpy as np

run_root = Path("workspace/runs/oracle_180709_E/validation_strel_fix_output_v28")
candidates_path = run_root / "02_Output" / "python_results" / "checkpoints" / "checkpoint_edge_candidates.pkl"

if not candidates_path.exists():
    print("Missing candidates file")
    exit(1)

candidates = joblib.load(candidates_path)
connections = candidates["connections"]

v_target = 92 # 0-indexed in Python
python_neighbors = []
for i in range(len(connections)):
    if connections[i, 0] == v_target:
        python_neighbors.append(int(connections[i, 1]))
    elif connections[i, 1] == v_target:
        python_neighbors.append(int(connections[i, 0]))

print(f"Vertex {v_target} Python neighbors: {sorted(python_neighbors)}")
