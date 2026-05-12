import joblib
import numpy as np
from slavv_python.core.global_watershed import _generate_edge_candidates_matlab_global_watershed
from slavv_python.core.energy_config import _matlab_lumen_radius_range
import json

# Load data from our verified synthetic container
v_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")
e_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl")

v_pos = v_data['positions']
v_scales = v_data['scales']
energy = e_data['energy']

# Params from the real run config
with open(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\final_audit_destination\01_Params\shared_params.json", "r") as f:
    params = json.load(f)

# CALIBRATED LUMEN RADIUS RANGE IN MICRONS
# Re-derive precisely as the runtime did
_, lumen_radius_microns = _matlab_lumen_radius_range(
    radius_smallest=1.5,
    radius_largest=60.0,
    scales_per_octave=6.0
)
lumen_radius_microns = lumen_radius_microns.flatten()

# THE FIX! Re-align microns to align with the volume's [Z, Y, X] data storage order
# Original passed was [0.916, 0.916, 1.996]
aligned_microns = np.array([1.99688, 0.916, 0.916], dtype=np.float32)
print(f"APPLYING ALIGNED MICRONS: {aligned_microns}")

# Execute the pure algorithm generator directly using EXACT inputs
results = _generate_edge_candidates_matlab_global_watershed(
    energy=energy,
    scale_indices=e_data.get('scale_indices'),
    vertex_positions=v_pos,
    vertex_scales=v_scales,
    lumen_radius_microns=lumen_radius_microns,
    microns_per_voxel=aligned_microns, # THE NEW VECTOR!
    _vertex_center_image=None,
    params=params
)

p_connections = results['connections']
print(f"GENERATED {len(p_connections)} PYTHON EDGES WITH ALIGNED MICRONS!")

# Now immediately audit against Oracle
oracle_edges = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\edges.pkl")
o_connections = oracle_edges['connections']

# Normalize and count matching pairs
o_set = set()
for a, b in o_connections:
    a, b = int(a), int(b)
    o_set.add(tuple(sorted((a, b))))

p_matches = 0
for a, b in p_connections:
    a, b = int(a), int(b)
    pair = tuple(sorted((a, b)))
    if pair in o_set:
        p_matches += 1

pct = (p_matches / len(o_set)) * 100 if len(o_set) > 0 else 0
print(f"\n=== ULTIMATE RESULTS ===")
print(f"ORACLE TOTAL EDGES   : {len(o_set)}")
print(f"PYTHON TOTAL EDGES   : {len(p_connections)}")
print(f"EXACT MATCHES FOUND  : {p_matches}")
print(f"RE-CALCULATED PARITY : {pct:.2f}%")
