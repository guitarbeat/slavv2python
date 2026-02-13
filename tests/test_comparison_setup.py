#!/usr/bin/env python3
"""
Quick test to verify comparison setup is correct.
This doesn't run the full vectorization, just checks that files exist and paths are correct.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_setup():
    """Test that all required files and paths exist."""
    print("Testing comparison setup...")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # Check test data file
    test_data = project_root / "data" / "slavv_test_volume.tif"
    if test_data.exists():
        print(f"[OK] Test data file found: {test_data}")
    else:
        errors.append(f"[ERROR] Test data file not found: {test_data}")
    
    # Check MATLAB executable
    matlab_path = Path("C:/Program Files/MATLAB/R2019a/bin/matlab.exe")
    if matlab_path.exists():
        print(f"[OK] MATLAB executable found: {matlab_path}")
    else:
        warnings.append(f"[WARN] MATLAB executable not found at default path: {matlab_path}")
        print(f"  (You can specify a different path with --matlab-path)")
    
    # Check scripts
    scripts_dir = project_root / "scripts"
    required_scripts = [
        "run_matlab_vectorization.m",
        "run_matlab_cli.bat",
        "compare_matlab_python.py",
        "comparison_params.json"
    ]
    
    print(f"\nChecking scripts in {scripts_dir}:")
    for script in required_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"  [OK] {script}")
        else:
            errors.append(f"  [ERROR] {script} not found")
    
    # Check Vectorization-Public
    matlab_repo_path = project_root / 'external' / 'Vectorization-Public'
    if matlab_repo_path.exists():
        print(f"\n[OK] Vectorization-Public directory found: {matlab_repo_path}")
        
        # Check for vectorize_V200.m
        vectorize_file = matlab_repo_path / "vectorize_V200.m"
        if vectorize_file.exists():
            print(f"  [OK] vectorize_V200.m found")
        else:
            errors.append(f"  [ERROR] vectorize_V200.m not found in Vectorization-Public")
    else:
        errors.append(f"[ERROR] Vectorization-Public directory not found: {matlab_repo_path}")
    
    # Check Python imports
    print(f"\nChecking Python imports:")
    try:
        from slavv.core import SLAVVProcessor
        print("  [OK] SLAVVProcessor imported successfully")
    except ImportError as e:
        errors.append(f"  [ERROR] Failed to import SLAVVProcessor: {e}")
    
    try:
        from slavv.io import load_tiff_volume
        print("  [OK] load_tiff_volume imported successfully")
    except ImportError as e:
        errors.append(f"  [ERROR] Failed to import load_tiff_volume: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print("ERRORS FOUND:")
        for error in errors:
            print(f"  {error}")
        print("\nPlease fix these errors before running the comparison.")
        return False
    else:
        print("[OK] All checks passed!")
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  {warning}")
        print("\nSetup is ready. You can now run:")
        print(f'  python scripts/compare_matlab_python.py \\')
        print(f'      --input "data/slavv_test_volume.tif" \\')
        print(f'      --matlab-path "C:\\Program Files\\MATLAB\\R2019a\\bin\\matlab.exe" \\')
        print(f'      --output-dir "comparison_output"')
        return True

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
