
import sys
import os
import json
import numpy as np
import copy
from pathlib import Path

# Add source to path
sys.path.append(os.getcwd())

def test_json_serialization():
    print("Testing JSON serialization with numpy arrays...")
    data = {
        'array': np.array([1, 2, 3]),
        'float': np.float64(1.23),
        'nested': {
            'list_of_arrays': [np.array([4, 5]), np.array([6])],
            'tuple': (1, 2)
        }
    }
    
    try:
        # Simulate the logic in comparison.py
        # report = {'data': copy.deepcopy(data)} # deepcopy might fail on some numpy objects? No, typically fine.
        
        # We need to simulate exactly what comparison.py does:
        # json.dump(report, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
        
        json_str = json.dumps(data, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
        print("Serialization successful!")
        print(json_str)
        return True
    except Exception as e:
        print(f"Serialization failed: {e}")
        return False

def test_module_import():
    print("\nTesting module import...")
    try:
        from source.slavv.dev import comparison
        print("Import successful.")
        return True
    except Exception as e:
        print(f"Import failed: {e}")
        return False

if __name__ == "__main__":
    success = True
    success &= test_json_serialization()
    success &= test_module_import()
    
    if success:
        print("\nALL VERIFICATIONS PASSED")
        sys.exit(0)
    else:
        print("\nVERIFICATION FAILED")
        sys.exit(1)
