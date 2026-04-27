import numpy as np
def check_view(dtype, order):
    a = np.zeros((10, 10, 10), dtype=dtype, order=order)
    b = a.ravel(order="F")
    print(f"Dtype: {dtype}, Order: {order}, Shares: {np.shares_memory(a, b)}")
    b[0] = 5
    if a[0,0,0] != 5:
        print("  FAILED to share data!")

check_view(np.uint64, "F")
check_view(np.uint32, "F")
check_view(np.float32, "F")
check_view(np.float64, "F")
check_view(np.int16, "F")
check_view(np.uint8, "F")
