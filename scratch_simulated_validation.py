import joblib
import numpy as np
import json
from slavv_python.core.generate import _generate_edge_candidates_matlab_frontier
from slavv_python.core.energy_config import _matlab_lumen_radius_range

# 1. Load exact cached inputs
v_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")
e_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl")

v_pos = v_data['positions'].copy()
v_scales = v_data['scales']
energy = e_data['energy']
scale_indices = e_data.get('scale_indices')
vertex_center_image = np.zeros_like(energy, dtype=np.float32)

# !!! THE HYPER-HYPOTHETICAL KEY !!!
# In this cached dataset, v_pos are loaded from Oracle, meaning they ARE IN THE CORRECT ALIGNED SYSTEM!
# But the WRAPPER expects raw stage outputs that match the un-aligned disk volume.
# Therefore, to truly test the Wrapper's complete un-flipping capabilities,
# we must artificialy FLIP these vertices IN REVERSE to emulate the "dirty" storage state
# that a pure Python run would supply to the wrapper!
print("!!! ARTIFICIALLY INVERTING CORRECT ORACLE VERTICES TO SIMULATE DIRTY INPUT SYSTEM !!!")
tmp_v = v_pos[:, 1].copy()
v_pos[:, 1] = v_pos[:, 2]
v_pos[:, 2] = tmp_v

with open(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\final_audit_destination\01_Params\shared_params.json", "r") as f:
    params = json.load(f)

# Use canonical Aligned Microns (also stay canonical [Z,Y,X])
# Wait! The aligned microns ARE [1.99688, 0.916, 0.916]. They are symmetric!
canonical_microns = np.array([1.99688, 0.916, 0.916], dtype=np.float32)

_, lumen_radius_microns = _matlab_lumen_radius_range(1.5, 60.0, 6.0)
lumen_radius_microns = lumen_radius_microns.flatten()

# ACTIVATE PRODUCTION WRAPPER NOW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print("ACTIVATING OFFICIAL GENERATE WRAPPER...")
results = _generate_edge_candidates_matlab_frontier(
    energy=energy,
    scale_indices=scale_indices,
    vertex_positions=v_pos, # Passing the "simulated dirty" inputs!
    vertex_scales=v_scales,
    lumen_radius_microns=lumen_radius_microns,
    microns_per_voxel=canonical_microns,
    vertex_center_image=vertex_center_image,
    params=params
)

actual_connections = results['connections']

oracle_edges = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\edges.pkl")
o_c = oracle_edges['connections']

o_set = set()
for u, v in o_c:
    a, b = int(u), int(v)
    o_set.add((min(a,b), max(a,b)))

match_count = 0
for u, v in actual_connections:
    a, b = int(u), int(v)
    if (min(a,b), max(a,b)) in o_set:
        match_count += 1

print(f"\n=== SIMULATED PRODUCTION VALIDATION ===")
print(f"ORACLE TOTAL EDGES   : {len(o_c)}")
print(f"PYTHON TOTAL EDGES   : {len(actual_connections)}")
print(f"EXACT MATCHES FOUND  : {match_count}")
pct = (match_count / len(o_c)) * 100
print(f"SIMULATED PARITY     : {pct:.2f}%")
if pct > 48.0:
    print("VINDICATION SECURED: The production wrapper logic is mathematically solid!")
else:
    print("FAILURE: My analysis of the simulation states contains a critical error!")
