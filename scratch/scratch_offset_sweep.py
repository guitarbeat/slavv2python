import joblib
import numpy as np

v_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")
e_data = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\runs\exact_seed_trial\02_Output\python_results\checkpoints\checkpoint_energy.pkl")

v_pos_raw = np.rint(v_data['positions']).astype(int)
v_scales = v_data['scales']
scale_indices = e_data.get('scale_indices')

# We already established that (0, 2, 1) is the right transpose for the volume
permuted = np.transpose(scale_indices, (0, 2, 1))

offsets = [-1, 0, 1]
for ox in offsets:
    for oy in offsets:
        for oz in offsets:
            diffs = []
            for i in range(len(v_pos_raw)):
                z, y, x = v_pos_raw[i]
                pz = z + oz
                py = y + oy
                px = x + ox
                
                if 0 <= pz < permuted.shape[0] and 0 <= py < permuted.shape[1] and 0 <= px < permuted.shape[2]:
                    val = permuted[pz, py, px]
                    diffs.append(abs(val - v_scales[i]))
            
            if diffs:
                avg_diff = np.mean(diffs)
                if avg_diff < 10:
                    print(f"OFFSET [Z={oz}, Y={oy}, X={ox}]: AVG DIFF = {avg_diff:.3f}")
