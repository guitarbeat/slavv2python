"""
Validation utilities for SLAVV comparison setup.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# Add project root to path if needed for imports
# (Assuming this module is importable as src.slavv.dev.validation)


class ValidationError:
    """Container for validation errors and warnings."""
    
    def __init__(self, message: str, level: str = "ERROR"):
        self.message = message
        self.level = level  # "ERROR", "WARNING", or "INFO"
    
    def __str__(self):
        return f"[{self.level}] {self.message}"


class Validator:
    """Validation runner for SLAVV comparison setup."""
    
    def __init__(self, project_root: Path, matlab_path: str = None, test_data_path: str = None):
        self.project_root = project_root
        self.matlab_path = matlab_path
        self.test_data_path = test_data_path
        self.errors = []
        self.warnings = []
        self.passed = []
        
    def add_error(self, message: str):
        self.errors.append(ValidationError(message, "ERROR"))
    
    def add_warning(self, message: str):
        self.warnings.append(ValidationError(message, "WARNING"))
    
    def add_pass(self, message: str):
        self.passed.append(ValidationError(message, "OK"))
    
    def check_file_exists(self, file_path: Path, description: str) -> bool:
        if file_path.exists():
            self.add_pass(f"{description}: {file_path}")
            return True
        else:
            self.add_error(f"{description} not found: {file_path}")
            return False
    
    def check_directory_exists(self, dir_path: Path, description: str) -> bool:
        if dir_path.exists() and dir_path.is_dir():
            self.add_pass(f"{description}: {dir_path}")
            return True
        else:
            self.add_error(f"{description} not found: {dir_path}")
            return False
    
    def check_python_imports(self) -> bool:
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
                self.add_error(f"Python package {display_name} is not installed")
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
        if not self.matlab_path:
            self.add_warning("No MATLAB path specified")
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
        if not self.test_data_path:
            self.add_warning("No test data path specified")
            return False
        
        data_path = Path(self.test_data_path)
        if not data_path.exists():
            self.add_error(f"Test data file not found: {data_path}")
            return False
        
        self.add_pass(f"Test data file exists: {data_path}")
        
        try:
            from src.slavv.io_utils import load_tiff_volume
            volume = load_tiff_volume(str(data_path))
            self.add_pass(f"Test data loaded successfully: shape={volume.shape}, dtype={volume.dtype}")
            
            if volume.ndim != 3:
                self.add_error(f"Test data should be 3D, got {volume.ndim}D")
                return False
                
            if volume.size == 0:
                self.add_error("Test data is empty")
                return False
            
            return True
        except Exception as e:
            self.add_error(f"Failed to load test data: {e}")
            return False

    def check_disk_space(self, output_dir: Path, required_gb: float = 5.0) -> bool:
        try:
            target = output_dir if output_dir.exists() else output_dir.parent
            stat = shutil.disk_usage(target)
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

    def check_vectorization_public(self) -> bool:
        matlab_repo_path = self.project_root / 'legacy' / 'Vectorization-Public' # Or external/Vectorization-Public depending on move
        # Check external first as that was the move target
        external_repo_path = self.project_root / 'external' / 'Vectorization-Public'
        
        repo_path = external_repo_path if external_repo_path.exists() else matlab_repo_path
        
        if not self.check_directory_exists(repo_path, "Vectorization-Public directory"):
            return False
            
        vectorize_file = repo_path / "vectorize_V200.m"
        if not self.check_file_exists(vectorize_file, "vectorize_V200.m"):
            return False
            
        source_dir = repo_path / "source"
        if not source_dir.exists():
            self.add_warning(f"'source' directory not found in Vectorization-Public. MATLAB may fail.")
        else:
            self.add_pass("Vectorization-Public/source directory found")
            
        return True
