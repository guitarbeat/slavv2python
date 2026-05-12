import joblib
import numpy as np

oracle_dir = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle"
oracle_edges_path = oracle_dir + r"\edges.pkl"

python_checkpoint_dir = r"d:\2P_Data\Aaron\slavv2python\workspace\runs\final_audit_destination\02_Output\python_results\checkpoints"
python_edges_path = python_checkpoint_dir + r"\checkpoint_edge_candidates.pkl"

o_edges = joblib.load(oracle_edges_path)
p_edges = joblib.load(python_edges_path)

o_c = o_edges['connections']
o_t = o_edges['traces']

p_c = p_edges['connections']
p_t = p_edges['traces']

# Build connection lookup by start index
o_lookup = {}
for i, (a, b) in enumerate(o_c):
    start = int(a)
    dest = int(b)
    if start not in o_lookup: o_lookup[start] = []
    o_lookup[start].append((dest, i))

# Find divergence
print("SCANNING FOR DIVERGENT PATHS STARTING FROM SAME VERTEX...")
found_divergence = False

for p_idx, (pa, pb) in enumerate(p_c):
    p_start = int(pa)
    p_dest = int(pb)
    
    if p_start in o_lookup:
        # Find if Oracle also traced from this start
        for o_dest, o_trace_idx in o_lookup[p_start]:
            if p_dest != o_dest:
                # DIVERGENCE DETECTED!
                print(f"\n!!! DIVERGENCE FOUND !!!")
                print(f"START VERTEX INDEX: {p_start}")
                print(f"ORACLE DESTINATION : {o_dest}")
                print(f"PYTHON DESTINATION : {p_dest}")
                
                p_path = p_t[p_idx]
                o_path = o_t[o_trace_idx]
                
                print(f"\nORACLE PATH LENGTH: {len(o_path)} waypoints")
                print(f"PYTHON PATH LENGTH: {len(p_path)} waypoints")
                
                print("\nWAYPOINT COMPARISON (INDEX | ORACLE XYZ | PYTHON XYZ | DIFF):")
                
                # Print top 20 waypoints
                limit = min(len(o_path), len(p_path), 20)
                for w in range(limit):
                    ow = o_path[w]
                    pw = p_path[w]
                    diff = np.linalg.norm(ow - pw)
                    print(f"{w:3d} | {ow} | {pw} | {diff:.6f}")
                    if diff > 0.01:
                        print(f"    --> SIGNIFICANT DEVIATION DETECTED AT STEP {w}")
                
                found_divergence = True
                break
        if found_divergence: break

if not found_divergence:
    print("No directly comparable start-node divergent pairs found.")
