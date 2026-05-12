import joblib
import numpy as np
from slavv_python.core.global_watershed import _generate_edge_candidates_matlab_global_watershed
from slavv_python.core.energy_config import _matlab_lumen_radius_range
import json

# Load data
v_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")
e_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl")

v_pos = v_data['positions']
v_scales = v_data['scales']
energy = e_data['energy']
scale_indices = e_data.get('scale_indices')

# APPLY UNIVERSE ROTATION
aligned_energy = np.transpose(energy, (0, 2, 1)).copy(order='F')
aligned_scale_indices = np.transpose(scale_indices, (0, 2, 1)).copy(order='F')

with open(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\final_audit_destination\01_Params\shared_params.json", "r") as f:
    params = json.load(f)

_, lumen_radius_microns = _matlab_lumen_radius_range(
    radius_smallest=1.5, radius_largest=60.0, scales_per_octave=6.0
)
lumen_radius_microns = lumen_radius_microns.flatten()

# USE THE ALIGNED MICRONS
aligned_microns = np.array([1.99688, 0.916, 0.916], dtype=np.float32)

print(f"RUNNING GENERATION TO EXTRACT NEW TRACES IN ALIGNED UNIVERSE...")
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
print("\nSCANNING FOR FIRST TRUE COSMIC DIVERGENCE:")
for i, (pa, pb) in enumerate(p_c):
    pa, pb = int(pa), int(pb)
    if pa in o_lookup:
        for ob, o_idx in o_lookup[pa]:
            if pb != ob:
                # BUT WAIT! Is this edge contained in oracle IN REVERSE or ELSEWHERE?
                # Let's verify if this SPECIFIC PAIR is entirely missing from Oracle
                # Check if (pa, pb) OR (pb, pa) is in ALL oracle edges
                found_anywhere = False
                for ou, ov in o_c:
                    if (int(ou) == pa and int(ov) == pb) or (int(ou) == pb and int(ov) == pa):
                        found_anywhere = True
                        break
                
                if not found_anywhere:
                    print(f"\n--- COSMIC DIVERGENCE {found_count+1} ---")
                    p0 = v_pos[pa].astype(int)
                    print(f"START VERTEX: {pa} at {p0}")
                    
                    # Python Step 1 Location
                    p_step_1 = p_t[i][1].astype(int)
                    
                    # Oracle might have multiple traces leaving this vertex. Let's look at ANY ONE.
                    o_step_1 = oracle_edges['traces'][o_idx][1].astype(int)
                    
                    print(f"PYTHON TARGETING TO {pb}")
                    print(f"PYTHON STEP 1: {p_step_1}")
                    print(f"ORACLE SAMPLE TRACE TO {ob}")
                    print(f"ORACLE STEP 1: {o_step_1}")
                    
                    # Print the deltas
                    p_d = p_step_1 - p0
                    o_d = o_step_1 - p0
                    print(f"PYTHON DELTA: {p_d}")
                    print(f"ORACLE DELTA: {o_d}")
                    
                    found_count += 1
                    break
        if found_count >= 1: break
