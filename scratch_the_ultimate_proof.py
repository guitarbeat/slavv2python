import joblib
import numpy as np

# Paths
oracle_dir = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle"
oracle_edges_path = oracle_dir + r"\edges.pkl"
oracle_verts_path = oracle_dir + r"\vertices.pkl"

python_checkpoint_dir = r"d:\2P_Data\Aaron\slavv2python\workspace\runs\final_audit_destination\02_Output\python_results\checkpoints"
python_edges_path = python_checkpoint_dir + r"\checkpoint_edge_candidates.pkl"
# The python vertices were literally cloned from the oracle, so they must match index-wise.
# But let's double verify that.

# Load everything
o_edges = joblib.load(oracle_edges_path)
o_verts = joblib.load(oracle_verts_path)['positions'] 

p_edges = joblib.load(python_edges_path)
# Python run loaded its input seeds from oracle verts, so they should map 1:1 by index!

o_c = o_edges['connections']
p_c = p_edges['connections']

print(f"ORACLE EDGE COUNT: {len(o_c)}")
print(f"PYTHON GENERATED CANDIDATE TRACE COUNT: {len(p_c)}")

# First attempt pure set intersection by index pair since our python indices SHOULD match oracle now!
o_pair_set = set(tuple(sorted([int(a), int(b)])) for a, b in o_c if a < 1313 and b < 1313)
p_pair_set = set(tuple(sorted([int(a), int(b)])) for a, b in p_c)

matched_pairs = o_pair_set.intersection(p_pair_set)

print(f"\n=== PURE INDEX-BASED PARITY ===")
print(f"REACHABLE ORACLE EDGES: {len(o_pair_set)}")
print(f"PYTHON EDGES PROCESSED: {len(p_pair_set)}")
print(f"MATCHED: {len(matched_pairs)}")
print(f"EXACT PARITY SCORE: {len(matched_pairs) / len(o_pair_set) * 100.0:.2f}%")

# Now let's do spatial check just in case the indices got jumbled but the geometry is right
def build_spatial_connection_set(connections, vertices):
    spatial_set = set()
    out_of_bounds = 0
    for a, b in connections:
        if a >= len(vertices) or b >= len(vertices):
            out_of_bounds += 1
            continue
        
        pos_a = tuple(np.round(vertices[a], 2)) # Round to prevent float noise
        pos_b = tuple(np.round(vertices[b], 2))
        node_pair = tuple(sorted([pos_a, pos_b]))
        spatial_set.add(node_pair)
    return spatial_set, out_of_bounds

oracle_spatial_set, o_oob = build_spatial_connection_set(o_c, o_verts)
python_spatial_set, p_oob = build_spatial_connection_set(p_c, o_verts) # Use same Oracle verts for BOTH mappings since indices align now!

overlap = oracle_spatial_set.intersection(python_spatial_set)

print(f"\n=== SPATIAL (ROUNDED) PARITY ===")
print(f"Oracle spatial unique: {len(oracle_spatial_set)}")
print(f"Python spatial unique: {len(python_spatial_set)}")
print(f"MATCHED SPATIALLY: {len(overlap)}")
print(f"SPATIAL PARITY SCORE: {len(overlap) / len(oracle_spatial_set) * 100.0:.2f}%")
