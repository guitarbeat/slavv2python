import numpy as np
a = np.zeros((3, 3, 3), order="F")
b = a.ravel(order="F")
print(f"Shares data: {np.shares_memory(a, b)}")
b[0] = 5
print(f"a[0,0,0]: {a[0,0,0]}")

c = np.zeros((3, 3, 3), order="C")
d = c.ravel(order="F")
print(f"C-order shares F-ravel data: {np.shares_memory(c, d)}")
d[0] = 5
print(f"c[0,0,0]: {c[0,0,0]}")
