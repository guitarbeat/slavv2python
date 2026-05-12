import joblib
import numpy as np

# Paths
oracle_dir = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle"
oracle_edges_path = oracle_dir + r"\edges.pkl"
oracle_verts_path = oracle_dir + r"\vertices.pkl"

python_checkpoint_dir = r"d:\2P_Data\Aaron\slavv2python\workspace\runs\revert_to_baseline_trial\02_Output\python_results\checkpoints"
python_edges_path = python_checkpoint_dir + r"\checkpoint_edge_candidates.pkl"
python_verts_path = python_checkpoint_dir + r"\checkpoint_vertices.pkl"

# Load everything
o_edges = joblib.load(oracle_edges_path)
o_verts = joblib.load(oracle_verts_path)['positions'] # shape (N, 3)

p_edges = joblib.load(python_edges_path)
p_verts = joblib.load(python_verts_path)['positions'] # shape (N, 3)

print(f"ORACLE VERTEX COUNT: {len(o_verts)}")
print(f"PYTHON VERTEX COUNT: {len(p_verts)}")

o_c = o_edges['connections']
p_c = p_edges['connections']

def build_spatial_connection_set(connections, vertices):
    spatial_set = set()
    out_of_bounds = 0
    for a, b in connections:
        if a >= len(vertices) or b >= len(vertices):
            out_of_bounds += 1
            continue
        
        pos_a = tuple(vertices[a])
        pos_b = tuple(vertices[b])
        # Sort nodes by coordinate tuple for deterministic ordering
        node_pair = tuple(sorted([pos_a, pos_b]))
        spatial_set.add(node_pair)
    return spatial_set, out_of_bounds

oracle_spatial_set, o_oob = build_spatial_connection_set(o_c, o_verts)
python_spatial_set, p_oob = build_spatial_connection_set(p_c, p_verts)

print(f"Oracle spatial connections: {len(oracle_spatial_set)} (Skipped {o_oob} out-of-bounds)")
print(f"Python spatial connections: {len(python_spatial_set)} (Skipped {p_oob} out-of-bounds)")

overlap = oracle_spatial_set.intersection(python_spatial_set)
matched_count = len(overlap)

print(f"MATCHED CONNECTIONS SPATIALLY: {matched_count}")
print(f"PARITY PERCENTAGE: {matched_count / len(oracle_spatial_set) * 100.0:.2f}%")
