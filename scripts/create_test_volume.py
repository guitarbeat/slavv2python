import os
import tifffile
import numpy as np
from slavv_python.utils.synthetic import generate_synthetic_vessel_volume

def main():
    os.makedirs("data", exist_ok=True)
    volume = generate_synthetic_vessel_volume(shape=(64, 64, 64), vessel_radius=5.0)
    # Ensure it's float32 for consistency
    volume = volume.astype(np.float32)
    tifffile.imwrite("data/slavv_test_volume.tif", volume)
    print("Created data/slavv_test_volume.tif")

if __name__ == "__main__":
    main()
