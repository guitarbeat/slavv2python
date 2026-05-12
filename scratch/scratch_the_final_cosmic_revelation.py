import joblib
import numpy as np
import json
from slavv_python.core.global_watershed import _generate_edge_candidates_matlab_global_watershed
from slavv_python.core.energy_config import _matlab_lumen_radius_range

# 1. Load exact cached inputs from the previous valid run
v_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")
e_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl")

v_pos = v_data['positions']
v_scales = v_data['scales']
energy = e_data['energy']
scale_indices = e_data.get('scale_indices')

# 2. APPLY THE UNIVERSE ROTATION!!! (0, 2, 1) TO ALIGN [Z, Y, X]
print("!!! APPLYING UNIVERSE ROTATION TRANSPOSE (0, 2, 1) !!!")
aligned_energy = np.transpose(energy, (0, 2, 1)).copy(order='F')
aligned_scale_indices = np.transpose(scale_indices, (0, 2, 1)).copy(order='F')

with open(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\final_audit_destination\01_Params\shared_params.json", "r") as f:
    params = json.load(f)

# Force our validated aligned microns
aligned_microns = np.array([1.99688, 0.916, 0.916], dtype=np.float32)
print(f"APPLYING ALIGNED MICRONS: {aligned_microns}")

_, lumen_radius_microns = _matlab_lumen_radius_range(
    radius_smallest=1.5, radius_largest=60.0, scales_per_octave=6.0
)
lumen_radius_microns = lumen_radius_microns.flatten()

results = _generate_edge_candidates_matlab_global_watershed(
    energy=aligned_energy,
    scale_indices=aligned_scale_indices,
    vertex_positions=v_pos,
    vertex_scales=v_scales,
    lumen_radius_microns=lumen_radius_microns,
    microns_per_voxel=aligned_microns,
    _vertex_center_image=None,
    params=params
)

py_c = results['connections']
oracle_edges = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\edges.pkl")
o_c = oracle_edges['connections']

o_set = set()
for u, v in o_c:
    a, b = int(u), int(v)
    o_set.add((min(a,b), max(a,b)))

match_count = 0
for u, v in py_c:
    a, b = int(u), int(v)
    if (min(a,b), max(a,b)) in o_set:
        match_count += 1

print(f"\n=== FINAL COSMIC REVELATION ===")
print(f"ORACLE TOTAL EDGES   : {len(o_c)}")
print(f"PYTHON TOTAL EDGES   : {len(py_c)}")
print(f"EXACT MATCHES FOUND  : {match_count}")
pct = (match_count / len(o_c)) * 100
print(f"ULTIMATE PARITY      : {pct:.2f}%")
