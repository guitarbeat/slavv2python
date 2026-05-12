import sys
from pathlib import Path
from scipy.io import loadmat
import numpy as np

batch_dir = Path(r"D:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\01_Input\matlab_results\batch_190910-103039_canonical")
net_path = list(batch_dir.glob("**/network_*.mat"))[0]

data = loadmat(net_path)
print(f"KEYS IN NETWORK FILE: {data.keys()}")
for k,v in data.items():
    if '__' not in k:
        try:
            if hasattr(v, 'shape'):
                print(f"  Key '{k}' Shape: {v.shape}")
        except:
            pass
