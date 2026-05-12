import sys
from pathlib import Path
from scipy.io import loadmat
import numpy as np

oracle_root = Path(r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039")
settings_dir = oracle_root / "01_Input" / "matlab_results" / "batch_190910-103039_canonical" / "settings"

print("SCANNING ALL MAT FILES IN SETTINGS FOR VERTEX COUNTS:")
files = sorted(settings_dir.glob("*.mat"))
for f in files:
    try:
        data = loadmat(f, squeeze_me=True, struct_as_record=False)
        v = None
        if hasattr(data, 'vertex_space_subscripts'):
            v = getattr(data, 'vertex_space_subscripts')
        elif 'vertex_space_subscripts' in data:
            v = data['vertex_space_subscripts']
        
        if v is not None:
            v_arr = np.atleast_2d(v)
            print(f"FILE: {f.name} -> COUNT: {len(v_arr)}")
    except Exception as e:
        pass
