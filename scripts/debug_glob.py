
from pathlib import Path
import os

batch_path = Path("comparisons/20260206_173559_matlab_run/matlab_results/batch_260206-173729")
vectors_dir = batch_path / 'vectors'

print(f"Vectors dir: {vectors_dir}")
print(f"Exists: {vectors_dir.exists()}")
print(f"Is dir: {vectors_dir.is_dir()}")

print("Listing dir:")
try:
    for f in vectors_dir.iterdir():
        print(f"  {f.name}")
except Exception as e:
    print(f"Error listing: {e}")

print("\nGlob vertices_*.mat:")
v_files = list(vectors_dir.glob('vertices_*.mat'))
print(f"Found: {len(v_files)}")
for f in v_files:
    print(f"  {f}")
