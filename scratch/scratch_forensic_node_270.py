import joblib
import numpy as np
import json
from slavv_python.core.common import _matlab_frontier_adjusted_neighbor_energies
from slavv_python.core.energy_config import _matlab_lumen_radius_range

v_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")
e_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl")

v_pos = v_data['positions']
v_scales = v_data['scales']
energy = e_data['energy']
scale_indices = e_data.get('scale_indices')

aligned_energy = np.transpose(energy, (0, 2, 1))
aligned_scale_indices = np.transpose(scale_indices, (0, 2, 1))

microns_per_voxel = np.array([1.99688, 0.916, 0.916], dtype=np.float32)

_, lumen_radius_microns = _matlab_lumen_radius_range(1.5, 60.0, 6.0)
lumen_radius_microns = lumen_radius_microns.flatten()

node_idx = 270
center = v_pos[node_idx].astype(int)
z0, y0, x0 = center

# Python picked [1, 0, 2] => [z0+1, y0+0, x0+2]
# Oracle picked [0, -1, 1] => [z0+0, y0-1, x0+1]
cand_py = np.array([1, 0, 2], dtype=int)
cand_or = np.array([0, -1, 1], dtype=int)

loc_py = center + cand_py
loc_or = center + cand_or

e_py = aligned_energy[loc_py[0], loc_py[1], loc_py[2]]
e_or = aligned_energy[loc_or[0], loc_or[1], loc_or[2]]

s_py = aligned_scale_indices[loc_py[0], loc_py[1], loc_py[2]]
s_or = aligned_scale_indices[loc_or[0], loc_or[1], loc_or[2]]

print(f"Node 270 Center: {center}")
print(f"Python Candidate {cand_py}: Raw Energy = {e_py:.6f}, Scale = {s_py}")
print(f"Oracle Candidate {cand_or}: Raw Energy = {e_or:.6f}, Scale = {s_or}")

# Recompute penalties for just these two
offsets = np.vstack([cand_py, cand_or])
vectors = offsets * microns_per_voxel
distances = np.linalg.norm(vectors, axis=1)

# Need the reference Radius R for the local strel based on vertex scale 
local_scale = aligned_scale_indices[z0, y0, x0]
R = lumen_radius_microns[int(local_scale)]
r_over_R = distances / R

print(f"Local Scale R = {R:.4f}")
print(f"Python Dist = {distances[0]:.4f}, r_over_R = {r_over_R[0]:.4f}")
print(f"Oracle Dist = {distances[1]:.4f}, r_over_R = {r_over_R[1]:.4f}")

raw_energies = np.array([e_py, e_or], dtype=np.float64)
scale_indices_cand = np.array([s_py, s_or], dtype=np.float64)

adj_energies = _matlab_frontier_adjusted_neighbor_energies(
    raw_energies=raw_energies,
    neighbor_offsets=offsets,
    neighbor_r_over_R=r_over_R,
    neighbor_scale_indices=scale_indices_cand,
    propagated_scale_index=int(local_scale),
    current_d_over_r=0.0, # Starting node
    origin_radius_microns=R,
    current_forward_unit=None, # First step has no forward direction
    microns_per_voxel=microns_per_voxel,
    lumen_radius_microns=lumen_radius_microns
)

print(f"ADJUSTED Python Energy: {adj_energies[0]:.8f}")
print(f"ADJUSTED Oracle Energy: {adj_energies[1]:.8f}")
