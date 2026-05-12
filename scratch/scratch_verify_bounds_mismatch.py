import joblib
import numpy as np

e_path = r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl"
v_path = r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl"

e_data = joblib.load(e_path)
v_data = joblib.load(v_path)

# Check keys
print(f"Energy Keys: {list(e_data.keys())}")
energy_vol = e_data['energy']

print(f"\nVOLUME SHAPE (RAW LOAD): {energy_vol.shape}")

v_pos = v_data['positions']
max_v = np.max(v_pos, axis=0)
min_v = np.min(v_pos, axis=0)

print(f"\nVERTEX MAX BOUNDS: {max_v}")
print(f"VERTEX MIN BOUNDS: {min_v}")

print(f"\nSAMPLE VERTEX (Index 1143): {v_pos[1143]}")
