import joblib
import numpy as np
import scipy.io
from slavv_python.core.global_watershed import _generate_edge_candidates_matlab_global_watershed

def run_reordered_test():
    energy_path = r"workspace\runs\measure2_experiment_evaluation\02_Output\python_results\checkpoints\checkpoint_energy.pkl"
    verts_path = r"workspace\runs\measure2_experiment_evaluation\02_Output\python_results\checkpoints\checkpoint_vertices.pkl"
    mat_path = r"workspace\oracles\180709_E_batch_190910-103039\01_Input\matlab_results\batch_190910-103039_canonical\vectors\curated_vertices_190910-172151_tie2gfp16 9juyly2018 870nm region a-082-1.mat"
    
    print("Loading energy...", flush=True)
    energy = joblib.load(energy_path)
    
    print("Loading Checkpoint Vertices...", flush=True)
    ckpt = joblib.load(verts_path)
    ckpt_pos = ckpt['positions'].astype(np.float32)
    ckpt_scales = ckpt['scales'].flatten().astype(np.int32)
    ckpt_energies = ckpt['energies'].flatten().astype(np.float32)
    
    print("Loading Raw MATLAB oracle to derive sequence...", flush=True)
    mat = scipy.io.loadmat(mat_path)
    raw_pos_mat = mat['vertex_space_subscripts'] # Matrix of [Y, X, Z]
    
    # The transformation derived earlier: Matlab [Y, X, Z] maps to Python [Z-1, X-1, Y-1]
    # Let's convert raw_pos_mat to the expected Python 0-indexed format
    raw_pos_converted = np.zeros((raw_pos_mat.shape[0], 3), dtype=np.float32)
    raw_pos_converted[:, 0] = raw_pos_mat[:, 2] - 1 # Z -> Python 0
    raw_pos_converted[:, 1] = raw_pos_mat[:, 1] - 1 # X -> Python 1
    raw_pos_converted[:, 2] = raw_pos_mat[:, 0] - 1 # Y -> Python 2
    
    print(f"Found {len(ckpt_pos)} vertices in checkpoint, and {len(raw_pos_converted)} in oracle mat.", flush=True)
    
    # Build a lookup to map converted coordinate strings to checkpoint indices
    # Round them to nearest int to tolerate precision diffs
    def to_key(p): return tuple(np.rint(p).astype(int).tolist())
    
    ckpt_lookup = {to_key(pos): idx for idx, pos in enumerate(ckpt_pos)}
    
    # Re-map to reconstruct sequence
    ordered_indices = []
    missing_count = 0
    for i, p in enumerate(raw_pos_converted):
        k = to_key(p)
        if k in ckpt_lookup:
            ordered_indices.append(ckpt_lookup[k])
        else:
            missing_count += 1
            
    print(f"Mapped {len(ordered_indices)} vertices successfully. Missing from checkpoint: {missing_count}", flush=True)
    
    # Re-order the checkpoint arrays!
    reordered_pos = ckpt_pos[ordered_indices]
    reordered_scales = ckpt_scales[ordered_indices]
    reordered_energies = ckpt_energies[ordered_indices]
    
    # We also need constants!
    # Typically derived from profile or loaded from params.
    lumen_radius_microns = np.asarray([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0], dtype=np.float32) # standard profile
    microns_per_voxel = np.asarray([1.0, 1.0, 1.0], dtype=np.float32) # will be overridden anyway?
    
    params = {
        "number_of_edges_per_vertex": 2,
        "energy_tolerance": 1.0,
        "radius_tolerance": 0.5,
        "step_size_per_origin_radius": 1.0,
        "distance_tolerance_per_origin_radius": 3.0,
    }
    
    shape = energy.shape
    vertex_center_image = np.zeros(shape, dtype=np.float32)
    
    print("Running MATLAB global watershed with RE-ORDERED sequence...", flush=True)
    def _hb(it, edges): print(f"  -> Iteration {it}, Edges so far: {edges}", flush=True)
    
    result = _generate_edge_candidates_matlab_global_watershed(
        energy=energy,
        scale_indices=None, # assume None or loaded from checkpoints
        vertex_positions=reordered_pos,
        vertex_scales=reordered_scales,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        _vertex_center_image=vertex_center_image,
        params=params,
        heartbeat=_hb
    )
    
    traces = result['traces']
    print(f"Generated {len(traces)} raw candidate traces!", flush=True)

if __name__ == "__main__":
    run_reordered_test()
