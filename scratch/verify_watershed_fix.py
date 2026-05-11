import os
import joblib
import numpy as np
from slavv_python.core.global_watershed import _generate_edge_candidates_matlab_global_watershed

def run_test():
    run_dir = r"workspace\runs\measure2_experiment_evaluation"
    energy_path = os.path.join(run_dir, r"02_Output\python_results\checkpoints\checkpoint_energy.pkl")
    vertex_path = os.path.join(run_dir, r"02_Output\python_results\checkpoints\checkpoint_vertices.pkl")
    
    print(f"Loading energy from {energy_path}")
    e_data = joblib.load(energy_path)
    energy = e_data['energy']
    scale_indices = e_data['scale_indices']
    lumen_radius_microns = e_data['lumen_radius_microns']
    
    print(f"Loading vertices from {vertex_path}")
    v_data = joblib.load(vertex_path)
    vertex_positions = v_data['positions']
    vertex_scales = v_data['scales']
    
    microns_per_voxel = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
    # Mock params
    params = {
        "edge_number_tolerance": 4,
        "energy_tolerance": 1.0,
        "distance_tolerance": 3.0,
        "radius_tolerance": 0.5,
        "step_size_per_origin_radius": 1.0
    }
    
    # We don't have vertex_center_image saved in checkpoints, but it's just a zero array if unused, 
    # or we can construct it easily.
    # Wait, the function calculates it or takes it as input?
    # Let's look at the function definition again. It takes vertex_center_image.
    shape = energy.shape
    vertex_center_image = np.zeros(shape, dtype=np.float32)
    
    print("Running MATLAB global watershed...", flush=True)
    print("Running MATLAB global watershed...", flush=True)
    def _hb(it, edges): print(f"  -> Iteration {it}, Edges so far: {edges}", flush=True)
    
    result = _generate_edge_candidates_matlab_global_watershed(
        energy=energy,
        scale_indices=scale_indices,
        vertex_positions=vertex_positions,
        vertex_scales=vertex_scales,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        _vertex_center_image=vertex_center_image,
        params=params,
        heartbeat=_hb
    )
    
    traces = result['traces']
    print(f"Generated {len(traces)} raw candidate traces!", flush=True)
    
    # traces_sets = [set(map(tuple, t)) for t in traces]
    # num_overlaps = 0
    # for i in range(len(traces_sets)):
    #     for j in range(i + 1, len(traces_sets)):
    #         intersect = traces_sets[i].intersection(traces_sets[j])
    #         if len(intersect) > 1:
    #             num_overlaps += 1
    # print(f"Found {num_overlaps} overlapping trace pairs.", flush=True)

if __name__ == "__main__":
    run_test()
