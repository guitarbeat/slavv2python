import joblib
import json

p_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl")
v_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")

# Load actual parameter values passed to the run
import pickle
with open(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\01_Input\execution_params.pkl", "rb") as f:
    params = pickle.load(f)

print(f"DATA SHAPE: {p_data['energy'].shape}")
print(f"PARAMS MICRONS_PER_VOXEL: {params['microns_per_voxel']}")
print(f"ORACLE MICRONS_PER_VOXEL USED IN RUN: {params.get('microns_per_voxel')}")

print(f"\nVERTEX MAX BOUNDS: {v_data['positions'].max(axis=0)}")
