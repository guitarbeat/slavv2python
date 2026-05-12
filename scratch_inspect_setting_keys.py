import sys
from pathlib import Path
from scipy.io import loadmat

f = r"d:\2P_Data\Aaron\slavv2python\workspace\oracles\180709_E_batch_190910-103039\01_Input\matlab_results\batch_190910-103039_canonical\settings\vertices_190910-173954.mat"
data = loadmat(f)
print(f"KEYS: {data.keys()}")
