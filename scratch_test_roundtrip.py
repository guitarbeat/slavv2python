import numpy as np

def _coord_to_matlab_linear_index(coord, shape):
    # Matches slavv_python exactly
    y, x, z = (int(value) for value in coord[:3])
    return int(y + x * shape[0] + z * shape[0] * shape[1])

def _matlab_linear_index_to_coord(index, shape):
    # Matches slavv_python exactly
    xy_plane = shape[0] * shape[1]
    z = index // xy_plane
    pos_xy = index - z * xy_plane
    x = pos_xy // shape[0]
    y = pos_xy - x * shape[0]
    return np.array([y, x, z], dtype=np.int32)

# Simulation with real data parameters
real_shape = (64, 512, 512)
seed_coord = np.array([42, 272, 176], dtype=np.int32)

print(f"SOURCE SEED: {seed_coord}")
print(f"VOLUME SHAPE: {real_shape}")

linear = _coord_to_matlab_linear_index(seed_coord, real_shape)
print(f"LINEAR INDEX COMPUTED: {linear}")

round_trip = _matlab_linear_index_to_coord(linear, real_shape)
print(f"ROUND TRIP RESULT: {round_trip}")

print(f"MATCH? {np.array_equal(seed_coord, round_trip)}")
