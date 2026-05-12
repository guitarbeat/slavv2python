import joblib
import numpy as np
from pathlib import Path

norm_dir = Path(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle")

for pkl_name in ["edges.pkl", "network.pkl"]:
    pkl_path = norm_dir / pkl_name
    if not pkl_path.exists():
        print(f"{pkl_name} DOES NOT EXIST.")
        continue
        
    print(f"\n--- INSPECTING {pkl_name} ---")
    data = joblib.load(pkl_path)
    print(f"TYPE: {type(data)}")
    if isinstance(data, dict):
        for k, v in data.items():
            if hasattr(v, "shape"):
                print(f"KEY: {k} -> SHAPE: {v.shape}")
            elif hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
                print(f"KEY: {k} -> LEN: {len(v)}")
            else:
                print(f"KEY: {k} -> VALUE SAMPLE: {str(v)[:50]}")
    else:
        print(f"RAW: {str(data)[:100]}")
