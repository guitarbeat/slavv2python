import joblib
import numpy as np
import json
from slavv_python.core.global_watershed import _generate_edge_candidates_matlab_global_watershed
from slavv_python.core.energy_config import _matlab_lumen_radius_range

# Load exact cached inputs
v_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")
e_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl")

v_pos = v_data['positions']
v_scales = v_data['scales']
energy = e_data['energy']
scale_indices = e_data.get('scale_indices')

# APPLY UNIVERSE ROTATION
aligned_energy = np.transpose(energy, (0, 2, 1)).copy(order='F')
aligned_scale_indices = np.transpose(scale_indices, (0, 2, 1)).copy(order='F')

# SORT VERTICES BY ENERGY TO EMULATE MATLAB'S EXPECTED INPUT PRE-CONDITION!!!
# 1. Sample energies at vertex locations in the aligned volume
v_pos_rounded = np.rint(v_pos).astype(int)
sampled_energies = []
for i in range(len(v_pos_rounded)):
    z, y, x = v_pos_rounded[i]
    val = aligned_energy[z, y, x]
    sampled_energies.append(val)

sampled_energies = np.array(sampled_energies)

# 2. Sort ascending (best energy first = smallest negative number)
sorted_idx = np.argsort(sampled_energies)

# Apply permutation to ALL vertex input buffers consistently
v_pos_sorted = v_pos[sorted_idx]
v_scales_sorted = v_scales[sorted_idx]

print(f"ENFORCING ENERGY SORT ORDER! Best energy now first: {sampled_energies[sorted_idx[0]]}")
print(f"Worst energy now last: {sampled_energies[sorted_idx[-1]]}")

with open(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\final_audit_destination\01_Params\shared_params.json", "r") as f:
    params = json.load(f)

aligned_microns = np.array([1.99688, 0.916, 0.916], dtype=np.float32)
print(f"APPLYING ALIGNED MICRONS: {aligned_microns}")

_, lumen_radius_microns = _matlab_lumen_radius_range(1.5, 60.0, 6.0)
lumen_radius_microns = lumen_radius_microns.flatten()

results = _generate_edge_candidates_matlab_global_watershed(
    energy=aligned_energy,
    scale_indices=aligned_scale_indices,
    vertex_positions=v_pos_sorted, # PASS SORTED
    vertex_scales=v_scales_sorted,  # PASS SORTED
    lumen_radius_microns=lumen_radius_microns,
    microns_per_voxel=aligned_microns,
    _vertex_center_image=None,
    params=params
)

py_c_raw = results['connections']
# The output connections use NEW local sorted indices! 
# We MUST map back to original vertex names/IDs to compare correctly against Oracle!
# The 'sorted_idx' array maps output_id -> input_id
# Wait, actually the returned IDs in Python are directly mapping to vertex_positions array index.
# So output ID `i` corresponds to `sorted_idx[i]` in original array.
actual_connections = []
for u, v in py_c_raw:
    orig_u = sorted_idx[int(u)]
    orig_v = sorted_idx[int(v)]
    actual_connections.append((orig_u, orig_v))

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

print(f"\n=== HYPER COSMIC REVELATION (WITH ENERGY SORT) ===")
print(f"ORACLE TOTAL EDGES   : {len(o_c)}")
print(f"PYTHON TOTAL EDGES   : {len(actual_connections)}")
print(f"EXACT MATCHES FOUND  : {match_count}")
pct = (match_count / len(o_c)) * 100
print(f"ULTIMATE PARITY      : {pct:.2f}%")
