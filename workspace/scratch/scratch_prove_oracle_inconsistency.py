import joblib
import numpy as np

edges_path = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\edges.pkl"
vertices_path = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\vertices.pkl"

edges = joblib.load(edges_path)
verts = joblib.load(vertices_path)

c = edges['connections']
max_idx = np.max(c)
print(f"MAX VERTEX INDEX IN EDGES.PKL CONNECTIONS: {max_idx}")
print(f"TOTAL VERTICES DEFINED IN VERTICES.PKL: {len(verts['positions'])}")
if max_idx >= len(verts['positions']):
    print("!!! CRITICAL DATA INCONSISTENCY PROVED !!!")
    print("The oracle's edge definitions refer to vertices that DO NOT EXIST in its vertex list.")
