import joblib
import numpy as np

oracle_edges = joblib.load(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\edges.pkl")
o_c = oracle_edges['connections']
o_t = oracle_edges['traces']
o_e = oracle_edges['energies']

found = False
for i, (u, v) in enumerate(o_c):
    if int(u) == 270 or int(v) == 270:
        print(f"\n--- Found Trace involving Node 270 (Index {i}) ---")
        print(f"Connection: {u} -> {v}")
        trace = o_t[i]
        engs = o_e[i]
        print(f"Trace Length: {len(trace)}")
        print(f"Oracle Step 0 pos: {trace[0].astype(int)} Eng: {engs:.6f}")
        # Since energies might be scalars (total trace energy), let's just print it
        print(f"Energy object type: {type(engs)}")
        print(f"Energy object value: {engs}")
        found = True

if not found:
    print("Node 270 not found in oracle connections.")
