import numpy as np
a = np.zeros((10, 10, 10), order="F")
b = a.copy()
c = b.ravel(order="F")
print(f"Shares memory: {np.shares_memory(b, c)}")
c[0] = 5
print(f"b[0,0,0]: {b[0,0,0]}")
