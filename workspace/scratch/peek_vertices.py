import joblib
import numpy as np
import scipy.io
from slavv_python.core.global_watershed import _generate_edge_candidates_matlab_global_watershed

def run_test():
    energy_path = r"workspace\runs\measure2_experiment_evaluation\02_Output\python_results\checkpoints\checkpoint_energy.pkl"
    mat_path = r"workspace\oracles\180709_E_batch_190910-103039\01_Input\matlab_results\batch_190910-103039_canonical\vectors\curated_vertices_190910-172151_tie2gfp16 9juyly2018 870nm region a-082-1.mat"
    
    print("Loading energy from checkpoint...", flush=True)
    energy = joblib.load(energy_path)
    
    print("Loading RAW vertices from MATLAB Oracle...", flush=True)
    mat = scipy.io.loadmat(mat_path)
    
    # Raw MATLAB arrays: (Y, X, Z) format, 1-indexed, float64/uint16
    raw_pos = mat['vertex_space_subscripts'].astype(np.float32)
    raw_scales = mat['vertex_scale_subscripts'].flatten().astype(np.int32)
    raw_energies = mat['vertex_energies'].flatten().astype(np.float32)
    
    # Coordinate transformation derived earlier: Python = [Z-1, X-1, Y-1]
    # Wait... in Python `slavv`, positions are typically stored in standard spatial coordinate format,
    # and are transformed during `_initialize_matlab_global_watershed_state`.
    # Let's load ONE position from `checkpoint_vertices.pkl` to confirm exactly what the format should be!
    pass

if __name__ == "__main__":
    # Read sample vertex first to confirm mapping
    import joblib
    d = joblib.load(r"workspace\runs\measure2_experiment_evaluation\02_Output\python_results\checkpoints\checkpoint_vertices.pkl")
    print("Checkpoint Position Sample:", d['vertex_positions'][0])
    print("Checkpoint Energy Sample:", d['vertex_energies'][0])
