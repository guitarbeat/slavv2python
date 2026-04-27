import numpy as np
a = np.zeros((10, 10, 10), order="F")
b = a.ravel(order="F")
b[0] = 1
print(f"a[0,0,0] = {a[0,0,0]}")
