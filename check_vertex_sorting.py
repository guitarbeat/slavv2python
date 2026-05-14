import joblib
import numpy as np
from pathlib import Path

data = joblib.load('workspace/runs/oracle_180709_E/validation_strel_fix/02_Output/python_results/checkpoints/checkpoint_vertices.pkl')
energies = data['energies']

# Sort indices by energy ascending (best first)
# Matches load_normalized_matlab_stage logic
sort_idx = np.argsort(energies, kind="stable")
sorted_energies = energies[sort_idx]

print("Top 15 best vertices (should be popped first):")
for i in range(15):
    idx = sort_idx[i]
    print(f"Vertex {idx}: Energy {energies[idx]}")

print("\nLast 15 vertices (should be popped last):")
for i in range(1, 16):
    idx = sort_idx[-i]
    print(f"Vertex {idx}: Energy {energies[idx]}")
