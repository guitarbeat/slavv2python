import joblib
import json

oracle_dir = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle"
v_path = oracle_dir + r"\vertices.pkl"
v_loaded = joblib.load(v_path)

print(f"ORACLE VERTEX POSITIONS MAX BOUNDS (Axis 0, 1, 2):")
print(v_loaded['positions'].max(axis=0))

with open(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\01_Params\shared_params.json", "r") as f:
    params = json.load(f)
    print(f"\nORACLE SOURCE MICRONS_PER_VOXEL: {params['microns_per_voxel']}")
