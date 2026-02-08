import time
import numpy as np
import sys
import os

# Ensure we can import slavv
sys.path.append(os.path.join(os.getcwd(), 'source'))

from slavv.core.tracing import trace_edge

def benchmark_trace_edge():
    # Setup
    shape = (100, 100, 100)
    # Energy sign is -1.0, so vessels are negative (minima)
    energy = -np.random.rand(*shape).astype(np.float64)

    # Create a valley along the diagonal
    for i in range(100):
        energy[i, i, i] = -10.0
        if i < 99:
            energy[i+1, i, i] = -8.0
            energy[i, i+1, i] = -8.0
            energy[i, i, i+1] = -8.0

    start_pos = np.array([10.0, 10.0, 10.0])
    direction = np.array([1.0, 1.0, 1.0])
    direction /= np.linalg.norm(direction)

    step_size = 0.5
    max_edge_energy = 0.0 # We stay below 0.0
    vertex_positions = np.empty((0, 3))
    vertex_scales = np.empty((0,), dtype=int)
    lumen_radius_pixels = np.array([1.0])
    lumen_radius_microns = np.array([1.0])
    max_steps = 500
    microns_per_voxel = np.array([1.0, 1.0, 1.0])
    energy_sign = -1.0

    # Warmup
    for _ in range(100):
        trace_edge(
            energy, start_pos, direction, step_size, max_edge_energy,
            vertex_positions, vertex_scales, lumen_radius_pixels, lumen_radius_microns,
            max_steps, microns_per_voxel, energy_sign
        )

    # Benchmark
    start_time = time.time()
    n_runs = 20000 # Increased by 10x
    total_steps = 0

    for _ in range(n_runs):
        trace = trace_edge(
            energy, start_pos, direction, step_size, max_edge_energy,
            vertex_positions, vertex_scales, lumen_radius_pixels, lumen_radius_microns,
            max_steps, microns_per_voxel, energy_sign
        )
        total_steps += len(trace)

    end_time = time.time()
    duration = end_time - start_time

    print(f"Time: {duration:.4f}s")
    print(f"Total steps: {total_steps}")
    print(f"Steps per second: {total_steps / duration:.2f}")

if __name__ == "__main__":
    benchmark_trace_edge()
