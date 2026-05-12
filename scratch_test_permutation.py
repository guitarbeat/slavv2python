import numpy as np

def _coord_to_matlab_linear_index(coord, shape):
    y, x, z = (int(value) for value in coord[:3])
    return int(y + x * shape[0] + z * shape[0] * shape[1])

def _matlab_linear_index_to_coord(index, shape):
    xy_plane = shape[0] * shape[1]
    z = index // xy_plane
    pos_xy = index - z * xy_plane
    x = pos_xy // shape[0]
    y = pos_xy - x * shape[0]
    return np.array([y, x, z], dtype=np.int32)

true_shape = (64, 512, 512)
swapped_shape = (512, 512, 64)

print("HYPOTHESIS TEST 1: What if input node [44, 269, 188] was packed with swap_shape?")
node_pos = np.array([44, 269, 188], dtype=np.int32)

# 1. Assume node_pos was originally intended for swapped_shape (where it's X,Y,Z and Z is 64)
# Wait, 188 is greater than 64, so it can't be the last dimension in swapped_shape.
# Okay, let's assume node_pos is [X, Y, Z] where X=44, Y=269, Z=188.
# Wait, Z can't be 188 if the Z-dimension is 64!!!

print("\nWait, let me test a different theory.")
print("What if the final coordinate output is swapped relative to the node?")
# Let's reverse-engineer the linear index of the recorded waypoint [63, 269, 44] assuming true_shape.
wp = np.array([63, 269, 44], dtype=np.int32)
linear = _coord_to_matlab_linear_index(wp, true_shape)
print(f"Linear index of waypoint [63, 269, 44] in shape (64, 512, 512) = {linear}")

# Now let's see if we can UNPACK this index using the OTHER shape (512, 512, 64)!!!
unpacked_swapped = _matlab_linear_index_to_coord(linear, swapped_shape)
print(f"Re-unpacked with shape (512, 512, 64) = {unpacked_swapped}")

print("\nWHAT ABOUT THE OTHER WAY AROUND?")
# Compute the linear index of node [44, 269, 188] in true shape
node_linear = _coord_to_matlab_linear_index(node_pos, true_shape)
print(f"Linear index of Node [44, 269, 188] in shape (64, 512, 512) = {node_linear}")

# Now re-unpack this using the OTHER shape (512, 512, 64)
unpacked_with_swapped = _matlab_linear_index_to_coord(node_linear, swapped_shape)
print(f"Re-unpacked with shape (512, 512, 64) = {unpacked_with_swapped}")
