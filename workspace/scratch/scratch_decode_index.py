import numpy as np
from slavv_python.core.global_watershed import _matlab_linear_index_to_coord

shape = (64, 512, 512)
val = 8145262

coord = _matlab_linear_index_to_coord(val, shape)
print(f"MYSTERY INDEX 8145262 MAPS TO COORD: {coord}")
