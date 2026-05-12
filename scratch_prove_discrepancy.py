import sys
from pathlib import Path
from scipy.io import loadmat
import numpy as np

oracle_root = Path(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039")
batch_dir = Path(r"D:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\01_Input\matlab_results\batch_190910-103039_canonical")

def count_v(path, label):
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    try:
        key = "vertex_space_subscripts"
        if hasattr(data, key):
            v = getattr(data, key)
        elif key in data:
            v = data[key]
        else:
            # Extract using common logic for struct as record=False
            v = getattr(data, key) # Should raise error if missing
        
        v_arr = np.atleast_2d(v)
        print(f"{label} Vertices count: {len(v_arr)}")
    except Exception as e:
        # Try another way, accessing dynamic dict
        for k, val in data.items():
             if 'vertex' in k.lower() and hasattr(val, 'shape') and len(val.shape)>0:
                 print(f"{label} Found potential '{k}' with len {len(val)}")
                 return
        print(f"{label} Failed: {e}")

cv_path = list(batch_dir.glob("**/curated_vertices_*.mat"))[0]
ed_path = list(batch_dir.glob("**/edges_*.mat"))[0]

print("LOADING CURATED VERTICES FILE:")
count_v(cv_path, "CURATED_VERTICES")

print("\nLOADING EDGES FILE:")
count_v(ed_path, "EDGES")

print("\nLOADING PRESERVED NORMALIZED EDGES PICKLE FOR ABSOLUTE TRUTH:")
import joblib
oracle_pickle = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\03_Analysis\normalized\oracle\edges.pkl"
data = joblib.load(oracle_pickle)
conn = np.asarray(data['connections'])
print(f"Max Vertex Index referenced in normalized pickle: {conn.max()}")
