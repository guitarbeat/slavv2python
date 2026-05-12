import scipy.io
import glob
import sys

def check_mat_types(file_path):
    print(f"Checking file: {file_path}")
    try:
        data = scipy.io.loadmat(file_path)
        for k, v in data.items():
            if hasattr(v, 'dtype'):
                print(f"  Variable '{k}': type={v.dtype}, shape={v.shape}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

files = glob.glob("workspace/oracles/180709_E_batch_190910-103039/01_Input/matlab_results/batch_190910-103039_canonical/vectors/*.mat")
if not files:
    print("No mat files found.")
else:
    check_mat_types(files[0])
