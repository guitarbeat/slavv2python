import numpy as np
import pytest
from slavv.core.tracing import _supplement_matlab_frontier_candidates_with_watershed_joins

def test_watershed_supplement_rejection_criteria():
    # Setup a simple 3D image with two vertices
    shape = (10, 10, 10)
    energy = np.zeros(shape, dtype=np.float32)
    
    # Create a "valley" of low energy between two points
    energy[2, 2, 2] = -10.0 # Vertex 0
    energy[2, 3, 2] = -8.0
    energy[2, 4, 2] = -8.0
    energy[2, 5, 2] = -8.0
    energy[2, 6, 2] = -10.0 # Vertex 1
    
    # Rest of the energy is higher (closer to 0)
    energy[energy == 0] = -1.0
    
    vertex_positions = np.array([
        [2.0, 2.0, 2.0],
        [2.0, 6.0, 2.0]
    ], dtype=np.float32)
    
    vertex_scales = np.array([0, 0], dtype=np.int32)
    
    candidates = {
        "connections": np.zeros((0, 2), dtype=np.int32),
        "traces": [],
        "metrics": [],
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": [],
        "connection_sources": [],
        "diagnostics": {}
    }
    
    # Call supplement. By default enforce_frontier_reachability=True
    # and since candidates["connections"] is empty, frontier_vertices will be empty.
    # So it should be rejected by reachability.
    result = _supplement_matlab_frontier_candidates_with_watershed_joins(
        candidates, energy, None, vertex_positions, energy_sign=-1.0,
        enforce_frontier_reachability=True
    )
    
    assert result["diagnostics"]["watershed_accepted"] == 0
    assert result["diagnostics"]["watershed_reachability_rejected"] > 0

def test_watershed_supplement_energy_rejection():
    # Setup a simple 3D image with two vertices
    shape = (10, 10, 10)
    # Background is higher than 0.0
    energy = -np.ones(shape, dtype=np.float32)
    
    # Create a full plane barrier of POSITIVE energy at Y=4
    # This ensures any path from Y < 4 to Y > 4 must cross Y=4.
    energy[:, 4, :] = 0.5 
    
    # Path along X=2, Z=2
    energy[2, :, 2] = -1.0 # foreground path
    energy[2, 4, 2] = 0.5    # background barrier in the path
    
    vertex_positions = np.array([
        [2.0, 2.0, 2.0],
        [2.0, 6.0, 2.0]
    ], dtype=np.float32)
    
    candidates = {
        "connections": np.zeros((0, 2), dtype=np.int32),
        "traces": [],
        "metrics": [],
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": [],
        "connection_sources": [],
        "diagnostics": {}
    }
    
    # Disable reachability so we can test energy rejection
    # We use energy_sign = 1.0 here (positive is foreground)
    # Background is lower than 0.0 (e.g. -1.0)
    energy = np.ones(shape, dtype=np.float32)
    energy[:, 4, :] = -0.5 # background barrier
    energy[2, 4, 2] = -0.5
    
    result = _supplement_matlab_frontier_candidates_with_watershed_joins(
        candidates, energy, None, vertex_positions, energy_sign=1.0,
        enforce_frontier_reachability=False
    )
    
    print(f"Energy at contact area: {energy[2, 2:7, 2]}")
    print(f"Diagnostics: {result['diagnostics']}")
    
    assert result["diagnostics"]["watershed_energy_rejected"] > 0
    assert result["diagnostics"]["watershed_accepted"] == 0

def test_watershed_supplement_energy_rejection_sign_pos():
    # Setup a simple 3D image with two vertices
    shape = (10, 10, 10)
    candidates = {
        "connections": np.zeros((0, 2), dtype=np.int32),
        "traces": [],
        "metrics": [],
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": [],
        "connection_sources": [],
        "diagnostics": {}
    }
    vertex_positions = np.array([
        [2.0, 2.0, 2.0],
        [2.0, 6.0, 2.0]
    ], dtype=np.float32)

    # We use energy_sign = 1.0 here (positive is foreground)
    # Background is lower than 0.0 (e.g. -1.0)
    energy = np.ones(shape, dtype=np.float32)
    energy[:, 4, :] = -0.5 # background barrier
    energy[2, 4, 2] = -0.5
    
    result = _supplement_matlab_frontier_candidates_with_watershed_joins(
        candidates, energy, None, vertex_positions, energy_sign=1.0,
        enforce_frontier_reachability=False
    )
    
    print(f"Energy at contact area (sign=1.0): {energy[2, 2:7, 2]}")
    print(f"Diagnostics: {result['diagnostics']}")
    
    assert result["diagnostics"]["watershed_energy_rejected"] > 0
    assert result["diagnostics"]["watershed_accepted"] == 0

if __name__ == "__main__":
    pytest.main([__file__])
