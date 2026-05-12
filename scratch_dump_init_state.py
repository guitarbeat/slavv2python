import joblib
import numpy as np
from slavv_python.core.global_watershed import _initialize_matlab_global_watershed_state, _coord_to_matlab_linear_index, _matlab_linear_index_to_coord

# Load verified data
v_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")
e_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl")

v_pos = v_data['positions']
energy = e_data['energy']
shape = energy.shape

print(f"Loaded data shapes: v={len(v_pos)}, energy={shape}")

target_idx = 1143
target_v = v_pos[target_idx]
print(f"TARGET VERTEX {target_idx} POSITION: {target_v}")

# Manually run initialization routine
state = _initialize_matlab_global_watershed_state(energy, v_pos)

v_locs = state['vertex_locations']
v_mapped = v_locs[target_idx]
print(f"STATE REVEAL: Vertex {target_idx} is mapped to LINEAR INDEX: {v_mapped}")

# Check mapped coordinate
c = _matlab_linear_index_to_coord(int(v_mapped), shape)
print(f"RE-EXPANDED COORDINATE FROM STATE: {c}")

# Inspect energy at state
print(f"ENERGY AT THIS LOCATION IN STATE MAP_TEMP: {state['energy_map_temp'][c[0], c[1], c[2]]}")

# Check neighbor generation for this precise location manually
from slavv_python.core.common import _build_matlab_global_watershed_lut
lut = _build_matlab_global_watershed_lut(
    scale_index=0,
    size_of_image=shape,
    lumen_radius_microns=e_data['lumen_radius_microns'],
    microns_per_voxel=np.array([0.916, 0.916, 1.996]), # Derived previously
    step_size_per_origin_radius=1.0
)

offsets = lut['linear_offsets']
print(f"\nSAMPLE NEIGHBOR OFFSETS FROM LUT: {offsets[:5]}")

# Print first 10 values of available_locations stack
avail = state['available_locations']
print(f"\nTOP 5 AVAILABLE LOCATIONS ON THE STACK (POPPED FIRST):")
for i in range(1, 6):
    loc = int(avail[-i])
    print(f"Stack position -{i}: Linear={loc}, Coord={_matlab_linear_index_to_coord(loc, shape)}")

print(f"\nBOTTOM 5 AVAILABLE LOCATIONS ON THE STACK:")
for i in range(5):
    loc = int(avail[i])
    print(f"Stack position {i}: Linear={loc}, Coord={_matlab_linear_index_to_coord(loc, shape)}")
