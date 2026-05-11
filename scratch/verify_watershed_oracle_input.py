import joblib
import numpy as np
import time
from slavv_python.core.global_watershed import _generate_edge_candidates_matlab_global_watershed

def run_oracle_input_test():
    energy_ckpt_path = r"workspace\runs\measure2_experiment_evaluation\02_Output\python_results\checkpoints\checkpoint_energy.pkl"
    oracle_verts_path = r"workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\vertices.pkl"
    
    print("Loading energy from checkpoint...", flush=True)
    energy_data = joblib.load(energy_ckpt_path)
    energy_vol = energy_data['energy']
    
    print("Loading Curated Oracle Vertices...", flush=True)
    oracle = joblib.load(oracle_verts_path)
    
    # The Oracle PKL has keys: 'positions', 'scales', 'energies'
    pos = oracle['positions'].astype(np.float32)
    scales = oracle['scales'].flatten().astype(np.int32)
    
    print(f"Loaded {len(pos)} curated vertices.", flush=True)
    
    lumen_radius_microns = np.asarray([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0], dtype=np.float32)
    microns_per_voxel = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    
    params = {
        "number_of_edges_per_vertex": 2,
        "energy_tolerance": 1.0,
        "radius_tolerance": 0.5,
        "step_size_per_origin_radius": 1.0,
        "distance_tolerance_per_origin_radius": 3.0,
    }
    
    shape = energy_vol.shape
    vertex_center_image = np.zeros(shape, dtype=np.float32)
    
    print("Running Python implementation on EXACT ORACLE INPUT...", flush=True)
    
    def _hb(it, edges):
        if it % 50 == 0:
            print(f"  -> Iteration {it}, Traces generated: {edges}", flush=True)
            
    start_t = time.time()
    result = _generate_edge_candidates_matlab_global_watershed(
        energy=energy_vol,
        scale_indices=None,
        vertex_positions=pos,
        vertex_scales=scales,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        _vertex_center_image=vertex_center_image,
        params=params,
        heartbeat=_hb
    )
    dur = time.time() - start_t
    
    traces = result['traces']
    print(f"\nSUCCESS! Run finished in {dur:.2f}s.", flush=True)
    print(f"Input Curated Vertices: {len(pos)}")
    print(f"Generated Candidate Traces: {len(traces)}")
    print("Expected Oracle Reference count was 1,197.", flush=True)

if __name__ == "__main__":
    run_oracle_input_test()
