#!/usr/bin/env python3
"""
Comprehensive pre-flight validation for MATLAB-Python comparison setup.

This script validates that all required files, paths, and dependencies are in place
before running the comparison. It can also test MATLAB execution with a minimal example.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ValidationError:
    """Container for validation errors and warnings."""
    
    def __init__(self, message: str, level: str = "ERROR"):
        self.message = message
        self.level = level  # "ERROR", "WARNING", or "INFO"
    
    def __str__(self):
        return f"[{self.level}] {self.message}"


class Validator:
    """Validation runner for MATLAB-Python comparison setup."""
    
    def __init__(self, matlab_path: str = None, test_data_path: str = None):
        self.matlab_path = matlab_path
        self.test_data_path = test_data_path
        self.errors = []
        self.warnings = []
        self.passed = []
        
    def add_error(self, message: str):
        """Add an error to the list."""
        self.errors.append(ValidationError(message, "ERROR"))
    
    def add_warning(self, message: str):
        """Add a warning to the list."""
        self.warnings.append(ValidationError(message, "WARNING"))
    
    def add_pass(self, message: str):
        """Add a passed check to the list."""
        self.passed.append(ValidationError(message, "OK"))
    
    def check_file_exists(self, file_path: Path, description: str) -> bool:
        """Check if a file exists."""
        if file_path.exists():
            self.add_pass(f"{description}: {file_path}")
            return True
        else:
            self.add_error(f"{description} not found: {file_path}")
            return False
    
    def check_directory_exists(self, dir_path: Path, description: str) -> bool:
        """Check if a directory exists."""
        if dir_path.exists() and dir_path.is_dir():
            self.add_pass(f"{description}: {dir_path}")
            return True
        else:
            self.add_error(f"{description} not found: {dir_path}")
            return False
    
    def check_python_imports(self) -> bool:
        """Check that all required Python packages can be imported."""
        required_packages = [
            ('numpy', 'NumPy'),
            ('scipy', 'SciPy'),
            ('tifffile', 'tifffile'),
            ('matplotlib', 'Matplotlib'),
        ]
        
        all_ok = True
        for module_name, display_name in required_packages:
            try:
                __import__(module_name)
                self.add_pass(f"Python package {display_name} is installed")
            except ImportError:
                self.add_error(f"Python package {display_name} is not installed. Install with: pip install {module_name}")
                all_ok = False
        
        # Check custom modules
        try:
            from src.slavv.pipeline import SLAVVProcessor
            self.add_pass("SLAVVProcessor can be imported")
        except ImportError as e:
            self.add_error(f"Cannot import SLAVVProcessor: {e}")
            all_ok = False
        
        try:
            from src.slavv.io_utils import load_tiff_volume
            self.add_pass("load_tiff_volume can be imported")
        except ImportError as e:
            self.add_error(f"Cannot import load_tiff_volume: {e}")
            all_ok = False
        
        return all_ok
    
    def check_matlab_executable(self) -> bool:
        """Check that MATLAB executable exists and is accessible."""
        if not self.matlab_path:
            self.add_warning("No MATLAB path specified. Use --matlab-path to validate MATLAB installation.")
            return False
        
        matlab_exe = Path(self.matlab_path)
        if not matlab_exe.exists():
            self.add_error(f"MATLAB executable not found: {matlab_exe}")
            return False
        
        self.add_pass(f"MATLAB executable found: {matlab_exe}")
        return True
    
    def check_matlab_version(self) -> bool:
        """Check MATLAB version and -batch flag support."""
        if not self.matlab_path:
            return False
        
        try:
            # Try to run MATLAB with -batch to check version
            result = subprocess.run(
                [self.matlab_path, '-batch', 'version; exit'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.add_pass("MATLAB -batch flag is supported (R2019a or later)")
                # Try to extract version from output
                if result.stdout:
                    for line in result.stdout.split('\n'):
                        if 'R20' in line:
                            self.add_pass(f"MATLAB version: {line.strip()}")
                            break
                return True
            else:
                self.add_warning("MATLAB -batch flag test failed. May need to use -r flag for older versions.")
                return False
        
        except subprocess.TimeoutExpired:
            self.add_error("MATLAB execution timed out. Check MATLAB installation.")
            return False
        except Exception as e:
            self.add_error(f"Failed to check MATLAB version: {e}")
            return False
    
    def check_test_data(self) -> bool:
        """Check test data file integrity."""
        if not self.test_data_path:
            self.add_warning("No test data path specified. Use --test-data to validate data file.")
            return False
        
        data_path = Path(self.test_data_path)
        if not data_path.exists():
            self.add_error(f"Test data file not found: {data_path}")
            return False
        
        self.add_pass(f"Test data file exists: {data_path}")
        
        # Try to load and validate the data
        try:
            from src.slavv.io_utils import load_tiff_volume
            
            volume = load_tiff_volume(str(data_path))
            self.add_pass(f"Test data loaded successfully: shape={volume.shape}, dtype={volume.dtype}")
            
            # Check for reasonable dimensions
            if volume.ndim != 3:
                self.add_error(f"Test data should be 3D, got {volume.ndim}D")
                return False
            
            if volume.size == 0:
                self.add_error("Test data is empty")
                return False
            
            # Check data type
            if volume.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
                self.add_warning(f"Unusual data type: {volume.dtype}")
            
            return True
            
        except Exception as e:
            self.add_error(f"Failed to load test data: {e}")
            return False
    
    def check_disk_space(self, output_dir: Path, required_gb: float = 5.0) -> bool:
        """Check available disk space."""
        try:
            stat = shutil.disk_usage(output_dir if output_dir.exists() else output_dir.parent)
            free_gb = stat.free / (1024 ** 3)
            
            if free_gb < required_gb:
                self.add_warning(f"Low disk space: {free_gb:.1f} GB available (recommended: {required_gb:.1f} GB)")
                return False
            else:
                self.add_pass(f"Sufficient disk space: {free_gb:.1f} GB available")
                return True
        except Exception as e:
            self.add_warning(f"Could not check disk space: {e}")
            return False
    
    def check_scripts(self) -> bool:
        """Check that all required scripts exist."""
        scripts_dir = project_root / "scripts"
        required_scripts = [
            ("run_matlab_vectorization.m", "MATLAB wrapper script"),
            ("run_matlab_cli.bat", "Windows batch script"),
            ("compare_matlab_python.py", "Comparison script"),
            ("matlab_output_parser.py", "MATLAB output parser"),
            ("comparison_params.json", "Parameter configuration")
        ]
        
        all_ok = True
        for script_name, description in required_scripts:
            script_path = scripts_dir / script_name
            if not self.check_file_exists(script_path, description):
                all_ok = False
        
        return all_ok
    
    def check_vectorization_public(self) -> bool:
        """Check Vectorization-Public directory and vectorize_V200.m."""
        vec_dir = project_root / "Vectorization-Public"
        
        if not self.check_directory_exists(vec_dir, "Vectorization-Public directory"):
            return False
        
        vectorize_file = vec_dir / "vectorize_V200.m"
        if not self.check_file_exists(vectorize_file, "vectorize_V200.m"):
            return False
        
        # Check for source directory
        source_dir = vec_dir / "source"
        if not source_dir.exists():
            self.add_warning(f"'source' directory not found in Vectorization-Public. MATLAB may fail.")
        else:
            self.add_pass("Vectorization-Public/source directory found")
        
        return True
    
    def run_matlab_minimal_test(self) -> bool:
        """Run a minimal MATLAB command to verify it works."""
        if not self.matlab_path:
            return False
        
        print("\nRunning minimal MATLAB test...")
        try:
            result = subprocess.run(
                [self.matlab_path, '-batch', 'disp("MATLAB test OK"); exit'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and "MATLAB test OK" in result.stdout:
                self.add_pass("MATLAB minimal execution test passed")
                return True
            else:
                self.add_error(f"MATLAB minimal test failed (exit code: {result.returncode})")
                if result.stderr:
                    print(f"  STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            self.add_error(f"MATLAB minimal test failed: {e}")
            return False
    
    def run_all_checks(self, minimal_matlab_test: bool = False) -> bool:
        """Run all validation checks.
        
        Parameters
        ----------
        minimal_matlab_test : bool
            If True, run a minimal MATLAB execution test
            
        Returns
        -------
        bool
            True if all critical checks passed
        """
        print("="*70)
        print("MATLAB-Python Comparison Setup Validation")
        print("="*70)
        
        print("\n--- Python Dependencies ---")
        self.check_python_imports()
        
        print("\n--- Required Scripts ---")
        self.check_scripts()
        
        print("\n--- Vectorization-Public Repository ---")
        self.check_vectorization_public()
        
        if self.matlab_path:
            print("\n--- MATLAB Installation ---")
            self.check_matlab_executable()
            self.check_matlab_version()
            
            if minimal_matlab_test:
                self.run_matlab_minimal_test()
        
        if self.test_data_path:
            print("\n--- Test Data ---")
            self.check_test_data()
        
        print("\n--- Disk Space ---")
        output_dir = project_root / "comparison_output"
        self.check_disk_space(output_dir)
        
        return self.print_summary()
    
    def print_summary(self) -> bool:
        """Print validation summary and return overall status."""
        print("\n" + "="*70)
        print("Validation Summary")
        print("="*70)
        
        if self.passed:
            print(f"\n✓ Passed Checks ({len(self.passed)}):")
            for item in self.passed[:10]:  # Show first 10
                print(f"  {item}")
            if len(self.passed) > 10:
                print(f"  ... and {len(self.passed) - 10} more")
        
        if self.warnings:
            print(f"\n⚠ Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.errors:
            print(f"\n✗ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  {error}")
            print("\n❌ VALIDATION FAILED")
            print("Please fix the errors above before running the comparison.")
            return False
        
        if self.warnings:
            print("\n⚠ VALIDATION PASSED WITH WARNINGS")
            print("The comparison may run, but some features may not work correctly.")
        else:
            print("\n✓ VALIDATION PASSED")
            print("All checks successful! You can proceed with the comparison.")
        
        print("\n" + "="*70)
        print("Next Steps:")
        print("="*70)
        
        # Provide usage instructions
        example_cmd = 'python scripts/compare_matlab_python.py \\\n'
        
        if self.test_data_path:
            example_cmd += f'    --input "{self.test_data_path}" \\\n'
        else:
            example_cmd += '    --input "data/slavv_test_volume.tif" \\\n'
        
        if self.matlab_path:
            example_cmd += f'    --matlab-path "{self.matlab_path}" \\\n'
        else:
            example_cmd += '    --matlab-path "C:\\Program Files\\MATLAB\\R2019a\\bin\\matlab.exe" \\\n'
        
        example_cmd += '    --output-dir "comparison_output"'
        
        print("\nRun the comparison with:")
        print(example_cmd)
        
        return len(self.errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Validate MATLAB-Python comparison setup',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--matlab-path',
        help='Path to MATLAB executable (e.g., C:\\Program Files\\MATLAB\\R2019a\\bin\\matlab.exe)'
    )
    parser.add_argument(
        '--test-data',
        help='Path to test data file for validation'
    )
    parser.add_argument(
        '--minimal-matlab-test',
        action='store_true',
        help='Run a minimal MATLAB execution test'
    )
    
    args = parser.parse_args()
    
    # Use default paths if not specified
    matlab_path = args.matlab_path or "C:\\Program Files\\MATLAB\\R2019a\\bin\\matlab.exe"
    test_data = args.test_data or str(project_root / "data" / "slavv_test_volume.tif")
    
    # Run validation
    validator = Validator(matlab_path=matlab_path, test_data_path=test_data)
    success = validator.run_all_checks(minimal_matlab_test=args.minimal_matlab_test)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
