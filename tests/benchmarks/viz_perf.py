
import time
import numpy as np
import plotly.graph_objects as go
from slavv.visualization import NetworkVisualizer

def generate_large_network(n_edges=5000):
    """Generates a large synthetic network."""
    vertices = {
        'positions': np.random.rand(n_edges * 2, 3) * 100,
        'energies': np.random.rand(n_edges * 2),
        'radii': np.random.rand(n_edges * 2) * 5,
        'radii_microns': np.random.rand(n_edges * 2) * 5
    }

    traces = []
    energies = []
    connections = []

    for i in range(n_edges):
        # Create a simple 2-point trace
        p1 = vertices['positions'][2*i]
        p2 = vertices['positions'][2*i+1]
        traces.append(np.array([p1, p2]))
        energies.append(np.random.rand())
        connections.append((2*i, 2*i+1))

    edges = {
        'traces': traces,
        'energies': np.array(energies),
        'connections': connections
    }

    network = {
        'bifurcations': [],
        'strands': []
    }

    parameters = {
        'microns_per_voxel': [1.0, 1.0, 1.0]
    }

    return vertices, edges, network, parameters

def benchmark_plot_2d_network():
    visualizer = NetworkVisualizer()
    vertices, edges, network, parameters = generate_large_network(n_edges=5000)

    print(f"[Benchmark]: Benchmarking plot_2d_network with {len(edges['traces'])} edges...")

    start_time = time.time()
    fig = visualizer.plot_2d_network(
        vertices, edges, network, parameters,
        color_by='energy',
        show_vertices=False, # Focus on edge rendering
        show_edges=True,
        show_bifurcations=False
    )
    end_time = time.time()

    print(f"[Benchmark]: Time taken: {end_time - start_time:.4f} seconds")
    print(f"[Benchmark]: Number of traces: {len(fig.data)}")

    # Verify trace count. Current implementation adds 1 trace per edge.
    # So we expect ~5000 traces.

if __name__ == "__main__":
    benchmark_plot_2d_network()
