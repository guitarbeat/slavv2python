import joblib
import numpy as np
from slavv_python.core.global_watershed import _generate_edge_candidates_matlab_global_watershed, _matlab_linear_index_to_coord
from slavv_python.core.energy_config import _matlab_lumen_radius_range
import json

# Load data
v_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")
e_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl")

v_pos = v_data['positions']
v_scales = v_data['scales']
energy = e_data['energy']
scale_indices = e_data.get('scale_indices')

with open(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\final_audit_destination\01_Params\shared_params.json", "r") as f:
    params = json.load(f)

_, lumen_radius_microns = _matlab_lumen_radius_range(
    radius_smallest=1.5, radius_largest=60.0, scales_per_octave=6.0
)
lumen_radius_microns = lumen_radius_microns.flatten()

# USE THE ALIGNED MICRONS
aligned_microns = np.array([1.99688, 0.916, 0.916], dtype=np.float32)

print(f"RUNNING GENERATION TO EXTRACT NEW TRACES...")
results = _generate_edge_candidates_matlab_global_watershed(
    energy=energy,
    scale_indices=scale_indices,
    vertex_positions=v_pos,
    vertex_scales=v_scales,
    lumen_radius_microns=lumen_radius_microns,
    microns_per_voxel=aligned_microns,
    _vertex_center_image=None,
    params=params
)

p_c = results['connections']
p_t = results['traces']

# AUDIT AGAINST ORACLE
oracle_edges = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\edges.pkl")
o_c = oracle_edges['connections']

o_lookup = {}
for i, (a, b) in enumerate(o_c):
    sa = int(a)
    if sa not in o_lookup: o_lookup[sa] = []
    o_lookup[sa].append((int(b), i))

found_count = 0
print("\nSCANNING FOR DIVERGENCES IN NEW ALIGNED RUN:")
for i, (pa, pb) in enumerate(p_c):
    pa, pb = int(pa), int(pb)
    if pa in o_lookup:
        for ob, o_idx in o_lookup[pa]:
            if pb != ob:
                print(f"\n--- DIVERGENCE {found_count+1} ---")
                p0 = v_pos[pa].astype(int)
                print(f"START VERTEX: {pa} at {p0}")
                
                v_scale = v_scales[pa]
                actual_val_scale = scale_indices[p0[0], p0[1], p0[2]]
                print(f"VERTEX SAVED SCALE: {v_scale}")
                print(f"VOLUME SCALE VALUE: {actual_val_scale}")
                print(f"LUMEN MICRONS AT SCALE {actual_val_scale}: {lumen_radius_microns[int(actual_val_scale)-1] if int(actual_val_scale)>0 else 'N/A'}")
                
                # Check Python Step 1 Location
                p_step_1 = p_t[i][1].astype(int)
                o_step_1 = oracle_edges['traces'][o_idx][1].astype(int)
                print(f"PYTHON STEP 1: {p_step_1}")
                print(f"PYTHON ENERGY AT STEP 1: {energy[p_step_1[0], p_step_1[1], p_step_1[2]]}")
                print(f"ORACLE STEP 1: {o_step_1}")
                print(f"ORACLE ENERGY AT STEP 1: {energy[o_step_1[0], o_step_1[1], o_step_1[2]]}")
                
                found_count += 1
                break
        if found_count >= 1: break
