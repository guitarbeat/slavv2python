#!/usr/bin/env python3
"""
Quick test to verify comparison setup is correct.
This doesn't run the full vectorization, just checks that files exist and paths are correct.
"""

import sys
from pathlib import Path

# Add project root and source to path for slavv imports
project_root = Path(__file__).parent.parent.parent
source_dir = project_root / "source"
sys.path.insert(0, str(source_dir))

def run_setup_check():
    """Test that all required files and paths exist."""
    print("Testing comparison setup...")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # Check test data file (may be in data/ or tests/data/)
    test_data = project_root / "data" / "slavv_test_volume.tif"
    if not test_data.exists():
        test_data = project_root / "tests" / "data" / "slavv_test_volume.tif"
    if test_data.exists():
        print(f"[OK] Test data file found: {test_data}")
    else:
        warnings.append("[WARN] Test data file not found (optional for comparison)")
    
    # Check MATLAB executable
    matlab_path = Path("C:/Program Files/MATLAB/R2019a/bin/matlab.exe")
    if matlab_path.exists():
        print(f"[OK] MATLAB executable found: {matlab_path}")
    else:
        warnings.append(f"[WARN] MATLAB executable not found at default path: {matlab_path}")
        print("  (You can specify a different path with --matlab-path)")
    
    # Check scripts (in scripts/cli)
    scripts_dir = project_root / "scripts" / "cli"
    required_scripts = [
        "run_matlab_vectorization.m",
        "run_matlab_cli.bat",
    ]
    
    print(f"\nChecking scripts in {scripts_dir}:")
    for script in required_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"  [OK] {script}")
        else:
            errors.append(f"  [ERROR] {script} not found")
    
    # Check Vectorization-Public (in external/ per README)
    matlab_repo_path = project_root / 'external' / 'Vectorization-Public'
    if matlab_repo_path.exists():
        print(f"\n[OK] Vectorization-Public directory found: {matlab_repo_path}")
        
        # Check for vectorize_V200.m
        vectorize_file = matlab_repo_path / "vectorize_V200.m"
        if vectorize_file.exists():
            print("  [OK] vectorize_V200.m found")
        else:
            # Downgrade to warning for CI/uninitialized submodules
            warnings.append("  [WARN] vectorize_V200.m not found in Vectorization-Public (submodule may be empty)")
    else:
        warnings.append(f"[WARN] Vectorization-Public directory not found: {matlab_repo_path} (required for MATLAB comparison)")
    
    # Check Python imports
    print("\nChecking Python imports:")
    try:
        from slavv.core import SLAVVProcessor  # noqa: F401
        print("  [OK] SLAVVProcessor imported successfully")
    except ImportError as e:
        errors.append(f"  [ERROR] Failed to import SLAVVProcessor: {e}")
    
    try:
        from slavv.io import load_tiff_volume  # noqa: F401
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
        print('  python scripts/cli/compare_matlab_python.py \\')
        print('      --input "data/slavv_test_volume.tif" \\')
        print('      --matlab-path "C:\\Program Files\\MATLAB\\R2019a\\bin\\matlab.exe" \\')
        print('      --output-dir "comparison_output"')
        return True

def test_setup_check():
    """Wrapper for pytest to assert success."""
    assert run_setup_check() is True

if __name__ == "__main__":
    success = run_setup_check()
    sys.exit(0 if success else 1)
