
import pytest
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'slavv-streamlit'))

from src.visualization import NetworkVisualizer

@pytest.fixture
def synthetic_data_large():
    n_vertices = 100
    n_edges = 200 # Above threshold of 100

    positions = np.random.rand(n_vertices, 3) * 100
    energies = np.random.rand(n_vertices)
    radii = np.random.rand(n_vertices)

    vertices = {
        'positions': positions,
        'energies': energies,
        'radii': radii,
        'radii_microns': radii
    }

    traces = []
    connections = []
    edge_energies = []

    for _ in range(n_edges):
        idx1 = np.random.randint(0, n_vertices)
        idx2 = np.random.randint(0, n_vertices)
        trace = np.array([positions[idx1], positions[idx2]])
        traces.append(trace)
        connections.append([idx1, idx2])
        edge_energies.append(np.random.rand())

    # Add a bifurcation manually
    network = {
        'bifurcations': [0, 1, 2],
        'strands': []
    }

    edges = {
        'traces': traces,
        'connections': connections,
        'energies': edge_energies
    }

    parameters = {
        'microns_per_voxel': [1.0, 1.0, 1.0]
    }

    return vertices, edges, network, parameters

@pytest.fixture
def synthetic_data_small():
    n_vertices = 20
    n_edges = 50 # Below threshold of 100

    positions = np.random.rand(n_vertices, 3) * 100
    energies = np.random.rand(n_vertices)
    radii = np.random.rand(n_vertices)

    vertices = {
        'positions': positions,
        'energies': energies,
        'radii': radii,
        'radii_microns': radii
    }

    traces = []
    connections = []
    edge_energies = []

    for _ in range(n_edges):
        idx1 = np.random.randint(0, n_vertices)
        idx2 = np.random.randint(0, n_vertices)
        trace = np.array([positions[idx1], positions[idx2]])
        traces.append(trace)
        connections.append([idx1, idx2])
        edge_energies.append(np.random.rand())

    edges = {
        'traces': traces,
        'connections': connections,
        'energies': edge_energies
    }

    network = {
        'bifurcations': [0, 1],
        'strands': []
    }

    parameters = {
        'microns_per_voxel': [1.0, 1.0, 1.0]
    }

    return vertices, edges, network, parameters

def test_plot_3d_network_merged_mode(synthetic_data_large):
    vertices, edges, network, parameters = synthetic_data_large
    viz = NetworkVisualizer()

    # Run with show_bifurcations=True to verify no regression
    fig = viz.plot_3d_network(vertices, edges, network, parameters, color_by='energy', show_bifurcations=True)

    # Expectation:
    # 1 trace for merged edges
    # 1 trace for vertices
    # 1 trace for bifurcations
    # Total traces should be small (<= 5)

    assert len(fig.data) <= 6, f"Expected few traces for merged mode, got {len(fig.data)}"

    merged_trace = None
    bif_trace = None
    for trace in fig.data:
        if trace.name == 'Network':
            merged_trace = trace
        if trace.name == 'Bifurcations':
            bif_trace = trace

    assert merged_trace is not None
    assert bif_trace is not None
    # Check point count
    assert len(merged_trace.x) >= 400

def test_plot_3d_network_legacy_mode(synthetic_data_small):
    vertices, edges, network, parameters = synthetic_data_small
    viz = NetworkVisualizer()

    # Run with show_bifurcations=True
    fig = viz.plot_3d_network(vertices, edges, network, parameters, color_by='energy', show_bifurcations=True)

    # Expectation:
    # 50 traces for edges (one per edge)
    # 1 trace for vertices
    # 1 trace for bifurcations
    # + colorbar traces
    # Total traces should be > 50

    assert len(fig.data) >= 50, f"Expected many traces for legacy mode, got {len(fig.data)}"

    bif_trace = None
    for trace in fig.data:
        if trace.name == 'Bifurcations':
            bif_trace = trace
            break

    assert bif_trace is not None
