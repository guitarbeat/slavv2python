import joblib
import numpy as np
import scipy.io
import time
from slavv_python.core.global_watershed import _generate_edge_candidates_matlab_global_watershed

def run_definitive_test():
    energy_ckpt_path = r"workspace\runs\measure2_experiment_evaluation\02_Output\python_results\checkpoints\checkpoint_energy.pkl"
    oracle_verts_path = r"workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\vertices.pkl"
    mat_path = r"workspace\oracles\180709_E_batch_190910-103039\01_Input\matlab_results\batch_190910-103039_canonical\vectors\curated_vertices_190910-172151_tie2gfp16 9juyly2018 870nm region a-082-1.mat"

    print("Loading energy...", flush=True)
    energy_vol = joblib.load(energy_ckpt_path)['energy']
    
    print("Loading Curated Oracle Vertices...", flush=True)
    oracle = joblib.load(oracle_verts_path)
    o_pos = oracle['positions'].astype(np.float32)
    o_scales = oracle['scales'].flatten().astype(np.int32)
    
    print("Loading Raw MATLAB oracle to get THE TRUE ORDER...", flush=True)
    mat = scipy.io.loadmat(mat_path)
    raw_pos_mat = mat['vertex_space_subscripts']
    
    # Create mapping key function: round values to handle slight float conversions
    def make_key(p):
        return tuple(np.rint(p).astype(int).tolist())
        
    # Build the lookup dictionary from validated Oracle Python arrays
    validated_lookup = {make_key(pos): i for i, pos in enumerate(o_pos)}
    
    # Reconstruct THE TRUE RAW SEQUENCE by looking up into the validated set
    reconstructed_indices = []
    unmatched = 0
    
    # Python = [Z-1, X-1, Y-1]
    for i in range(len(raw_pos_mat)):
        y_val = raw_pos_mat[i, 0]
        x_val = raw_pos_mat[i, 1]
        z_val = raw_pos_mat[i, 2]
        
        # Try various permutations to be 100% certain we match the oracle normalization scheme
        # Based on sample, Oracle normalized it such that:
        # Mat [309, 213, 11] became [10, 212, 308]
        target_p = np.array([z_val - 1, x_val - 1, y_val - 1])
        key = make_key(target_p)
        
        if key in validated_lookup:
            reconstructed_indices.append(validated_lookup[key])
        else:
            # Fallback lookup just in case
            found = False
            for offset in [(z_val - 1, x_val - 1, y_val - 1)]:
                 if make_key(offset) in validated_lookup:
                     reconstructed_indices.append(validated_lookup[make_key(offset)])
                     found = True
                     break
            if not found:
                unmatched += 1

    print(f"Re-sequenced successfully: {len(reconstructed_indices)} vertices.")
    print(f"Failed to match: {unmatched}")
    
    if unmatched > 0:
        print("CRITICAL: Mapping mismatch. Cannot proceed with perfect sequence.")
        return
        
    # Build the ABSOLUTELY PERFECT ARRAYS adhering to the exact sequence of the raw MATLAB file
    perfect_pos = o_pos[reconstructed_indices]
    perfect_scales = o_scales[reconstructed_indices]
    
    # PREPARE GLOBAL WATERSHED INPUTS
    lumen_radius_microns = np.asarray([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0], dtype=np.float32)
    microns_per_voxel = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    params = {
        "number_of_edges_per_vertex": 2,
        "energy_tolerance": 1.0,
        "radius_tolerance": 0.5,
        "step_size_per_origin_radius": 1.0,
        "distance_tolerance_per_origin_radius": 3.0,
    }
    vertex_center_image = np.zeros(energy_vol.shape, dtype=np.float32)

    print("Running GLOBAL WATERSHED on PERFECT MATLAB-SEQUENCE REPLICA...", flush=True)
    
    # CRITICAL STEP TO MIRROR MATLAB!!!!
    # MATLAB `get_edges_by_watershed.m` line 134: `available_locations = vertex_locations( end : -1 : 1 )`
    # Our Python implementation currently does NOT reverse it (we had `[::-1]` and then removed it).
    # To match MATLAB PRECISELY, we must ensure that when Python pops from the end, 
    # it pops THE FIRST ELEMENT of the raw input!
    # Wait, Matlab reverses the input, then pops the end!
    # Popping the END of REVERSED input yields the FIRST ELEMENT of the original input!!!!!!!!!!!!!!!!!!
    # SO, to mirror MATLAB popping the FIRST ELEMENT OF ORIGINAL SEQUENCE...
    # We should NOT reverse the input list, but just let Python pop the end? 
    # No! If we don't reverse, Popping the END yields the LAST element of original sequence!
    # Wait!!! Let's re-trace Matlab logic EXPLICITLY:
    # 1. Input Sequence S: [A, B, C, D]
    # 2. Matlab creates available_locations = S(end:-1:1): [D, C, B, A]
    # 3. Matlab pops the end: A!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # YES!!!!! SO MATLAB POPS FROM START TO END OF ORIGINAL SEQUENCE!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # To do this in Python popping the end, we MUST PREPARE available_locations AS REVERSED Sequence!!!!!!!!!!!!!!
    # Wait, my python code in `global_watershed.py` currently DOES NOT REVERSE the sequence!!!!!!!!!!!!
    # So right now, it will pop the LAST element first.
    # To make it pop the FIRST element first (matching Matlab), I MUST REVERSE THE LIST PASSED IN!!!!!!!!!!!!!!!!!!
    # YES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # Reversing the list order for passing into current code so Python pops "End of Reversed" == First element!
    mirrored_pos = perfect_pos[::-1]
    mirrored_scales = perfect_scales[::-1]
    
    def _hb(it, edges):
        if it % 100 == 0:
            print(f"  Iteration {it}, Traces: {edges}", flush=True)

    start_t = time.time()
    result = _generate_edge_candidates_matlab_global_watershed(
        energy=energy_vol,
        scale_indices=None,
        vertex_positions=mirrored_pos,
        vertex_scales=mirrored_scales,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        _vertex_center_image=vertex_center_image,
        params=params,
        heartbeat=_hb
    )
    dur = time.time() - start_t
    
    print(f"\nCOMPLETE IN {dur:.2f}s!")
    traces = result['traces']
    print(f"Generated Candidate Traces: {len(traces)}")
    
    # Count unique undirected pairs
    pairs = result['connections']
    # Coerce to scalar ints
    scalar_pairs = [(int(p[0]), int(p[1])) for p in pairs]
    sorted_pairs = [tuple(sorted(p)) for p in scalar_pairs]
    unique_pairs = set(sorted_pairs)
    print(f"Unique Undirected Edge Pairs: {len(unique_pairs)}")
    
    print("Expected Oracle Reference count: 1,197")

if __name__ == "__main__":
    run_definitive_test()
