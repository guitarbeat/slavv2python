import joblib
import numpy as np
from pathlib import Path

data = joblib.load('workspace/runs/oracle_180709_E/validation_strel_fix/02_Output/python_results/checkpoints/checkpoint_vertices.pkl')
p = data['positions']
r = data['radii']
v228_pos = p[228]

for i in range(len(p)):
    if i == 228: continue
    dist = np.linalg.norm(p[i] - v228_pos)
    if dist < 100:
        print(f"Vertex {i}: Pos={p[i]}, Radius={r[i]}, Dist={dist}")
