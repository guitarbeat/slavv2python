import joblib
import numpy as np
from itertools import permutations

v_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")
e_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl")

v_pos = np.rint(v_data['positions']).astype(int)
v_scales = v_data['scales']
scale_indices = e_data.get('scale_indices')

print(f"VOLUME SHAPE: {scale_indices.shape}")

# Verify permutations
valid_perms = []
# We know axis 0 must stay axis 0 because sizes are 64 vs 512
# Wait, is it possible axis 0 swapped with axis 1? No, bounds error will occur for indices > 64.
# But let's just test if axis 1 and 2 swapped: (0, 2, 1)
perms = [
    (0, 1, 2),
    (0, 2, 1)
]

for p in perms:
    permuted = np.transpose(scale_indices, p)
    diffs = []
    for i in range(len(v_pos)):
        z, y, x = v_pos[i]
        if 0 <= z < permuted.shape[0] and 0 <= y < permuted.shape[1] and 0 <= x < permuted.shape[2]:
            val = permuted[z, y, x]
            # The map values are stored as indices, but sometimes off-by-one depending on loading
            # Let's check proximity
            actual_scale = v_scales[i]
            diffs.append(abs(val - actual_scale))
    
    avg_diff = np.mean(diffs)
    print(f"PERMUTATION {p}: AVG DIFF = {avg_diff:.2f}")

# WAIT! What if Axis 0 IS NOT Z?!
# Let's just catch IndexError and test all 6 permutations!
all_perms = list(permutations([0, 1, 2]))
for p in all_perms:
    try:
        permuted = np.transpose(scale_indices, p)
        diffs = []
        for i in range(len(v_pos)):
            z, y, x = v_pos[i]
            # v_pos was assumed to be [Z, Y, X] which gave bounds valid for (64, 512, 512)
            val = permuted[z, y, x]
            actual_scale = v_scales[i]
            diffs.append(abs(val - actual_scale))
        avg_diff = np.mean(diffs)
        print(f"FULL TEST PERM {p}: AVG DIFF = {avg_diff:.2f}")
    except IndexError:
        print(f"FULL TEST PERM {p}: INDEX ERROR (Invalid bounds)")
        continue
