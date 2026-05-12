import joblib
import numpy as np

e_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl")
scale_indices = e_data.get('scale_indices')

z0, y0, x0 = 54, 375, 335
target_val = 43

print(f"SEARCHING FOR VALUE {target_val} IN NEIGHBORHOOD OF [{z0}, {y0}, {x0}]...")

min_dist = float('inf')
best_coord = None

for dz in range(-15, 16):
    for dy in range(-15, 16):
        for dx in range(-15, 16):
            z = z0 + dz
            y = y0 + dy
            x = x0 + dx
            if 0 <= z < 64 and 0 <= y < 512 and 0 <= x < 512:
                val = scale_indices[z, y, x]
                if val == target_val:
                    dist = (dz**2 + dy**2 + dx**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_coord = (z, y, x)

if best_coord:
    print(f"FOUND EXACT MATCH AT {best_coord} (Dist: {min_dist:.2f})")
else:
    print("NO MATCH FOUND IN 15-VOXEL CUBE. WIDENING SEARCH...")
    # Expand full spatial search for the closest pixel containing value 43 in the ENTIRE 3D SPACE
    matches = np.where(scale_indices == target_val)
    if len(matches[0]) > 0:
        coords = np.column_stack(matches)
        dists = np.sum((coords - [z0, y0, x0])**2, axis=1)**0.5
        idx = np.argmin(dists)
        print(f"CLOSEST GLOBAL MATCH AT {coords[idx]} (Dist: {dists[idx]:.2f})")
