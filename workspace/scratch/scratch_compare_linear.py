import numpy as np
from slavv_python.core.global_watershed import _coord_to_matlab_linear_index

shape = (64, 512, 512)
z, y, x = 54, 375, 335
coord = np.array([z, y, x], dtype=np.int32)

# 1. Calculate linear index exactly as the function does
linear_idx = _coord_to_matlab_linear_index(coord, shape)
print(f"LINEAR INDEX FOR [54, 375, 335]: {linear_idx}")

# 2. Compare against what WAS popped in the log
popped_val = 8145262
print(f"DID THEY MATCH? {linear_idx == popped_val}")

# 3. Let's see what coordinate YIELDS 8145262 using the opposite convention:
# (x, y, z) or (z, y, x)
# Function implements: val = y + x*64 + z*32768
print(f"PREVIOUS MANUAL CALCULATION: {54 + 375*64 + 335*32768}")
