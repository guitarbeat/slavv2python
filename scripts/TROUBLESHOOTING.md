# MATLAB-Python Comparison Troubleshooting Guide

This guide helps resolve common issues when running the MATLAB-Python comparison framework.

## Table of Contents

- [MATLAB Issues](#matlab-issues)
- [Python Issues](#python-issues)
- [Path Issues](#path-issues)
- [Output Issues](#output-issues)
- [Comparison Issues](#comparison-issues)
- [Performance Issues](#performance-issues)
- [Debugging Tips](#debugging-tips)

---

## MATLAB Issues

### MATLAB Not Found

**Symptom:** Error message "MATLAB executable not found"

**Solutions:**
1. Verify MATLAB installation path:
   ```bash
   # Check if MATLAB exists at the specified path
   "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" -batch "version; exit"
   ```

2. Use the correct path for your MATLAB version:
   - R2019a: `C:\Program Files\MATLAB\R2019a\bin\matlab.exe`
   - R2020b: `C:\Program Files\MATLAB\R2020b\bin\matlab.exe`
   - Adjust the path accordingly

3. Run validation script to check MATLAB:
   ```bash
   python scripts/validate_setup.py --matlab-path "C:\Path\To\matlab.exe"
   ```

### `-batch` Flag Not Supported

**Symptom:** MATLAB error about unrecognized `-batch` flag

**Cause:** MATLAB versions before R2019a don't support the `-batch` flag.

**Solutions:**
1. Upgrade to MATLAB R2019a or later (recommended)

2. For older MATLAB versions, modify `scripts/run_matlab_cli.bat`:
   - Replace `-batch` with `-r`
   - Change command format:
     ```batch
     REM Old (R2019a+):
     "%MATLAB_PATH%" -batch "%MATLAB_SCRIPT%"
     
     REM New (older versions):
     "%MATLAB_PATH%" -r "%MATLAB_SCRIPT%; exit"
     ```

3. Note: Older MATLAB versions may have different behavior or compatibility issues

### MATLAB Hangs or Crashes

**Symptom:** MATLAB execution never completes or crashes without error

**Possible Causes:**
- Out of memory
- Graphics/display issues
- License server problems
- Corrupted temporary files

**Solutions:**
1. Check available memory:
   ```bash
   # Windows Task Manager: Check available RAM
   # Ensure at least 8GB free for processing
   ```

2. Verify MATLAB runs interactively:
   ```matlab
   % Open MATLAB and test
   cd('legacy/Vectorization-Public')
   version
   ```

3. Check MATLAB license:
   ```matlab
   license('test', 'image_toolbox')
   ```

4. Clear MATLAB temporary files:
   ```batch
   REM Delete temp files in MATLAB prefdir
   matlab -batch "delete(fullfile(prefdir, '*.mat')); exit"
   ```

5. Reduce image size or increase timeout in comparison script

### vectorize_V200.m Not Found

**Symptom:** Error "Undefined function 'vectorize_V200'"

**Solutions:**
1. Verify Vectorization-Public directory exists in legacy:
   ```bash
   ls legacy/Vectorization-Public/vectorize_V200.m
   ```

2. Ensure the batch script changes to the correct directory:
   - Check `run_matlab_cli.bat` sets `VECTORIZATION_DIR` correctly
   - Verify the `cd` command executes before calling the function

3. Manually test MATLAB can find the function:
   ```matlab
   cd('C:\path\to\slavv2python\legacy\Vectorization-Public')
   which vectorize_V200
   % Should show: C:\path\to\slavv2python\legacy\Vectorization-Public\vectorize_V200.m
   ```

---

## Python Issues

### Import Errors

**Symptom:** `ImportError` or `ModuleNotFoundError`

**Common Missing Packages:**
```bash
pip install numpy scipy matplotlib tifffile
```

**Solutions:**
1. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Verify Python environment:
   ```python
   import numpy
   import scipy
   import tifffile
   print("All imports successful")
   ```

3. Check Python version (requires 3.8+):
   ```bash
   python --version
   ```

4. Use virtual environment to avoid conflicts:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

### SLAVVProcessor Import Error

**Symptom:** `Cannot import SLAVVProcessor`

**Solutions:**
1. Verify you're in the project root directory:
   ```bash
   cd /path/to/slavv2python
   python scripts/compare_matlab_python.py ...
   ```

2. Check PYTHONPATH includes project root:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/slavv2python"  # Linux/Mac
   set PYTHONPATH=%PYTHONPATH%;C:\path\to\slavv2python      # Windows
   ```

3. Verify src/slavv directory structure:
   ```bash
   ls src/slavv/pipeline.py
   ls src/slavv/__init__.py
   ```

### TIFF Loading Error

**Symptom:** Error loading TIFF file or "File is not a valid TIFF"

**Solutions:**
1. Verify TIFF file exists and is readable:
   ```python
   from pathlib import Path
   p = Path("data/slavv_test_volume.tif")
   print(f"Exists: {p.exists()}, Size: {p.stat().st_size}")
   ```

2. Check TIFF file integrity:
   ```python
   import tifffile
   volume = tifffile.imread("data/slavv_test_volume.tif")
   print(f"Shape: {volume.shape}, Dtype: {volume.dtype}")
   ```

3. Ensure file is 3D grayscale:
   - Should have shape (Z, Y, X)
   - Should be uint8, uint16, or float32

---

## Path Issues

### Spaces in Paths

**Symptom:** Errors with file paths containing spaces

**Solutions:**
1. Always quote paths with spaces:
   ```bash
   # Good
   python scripts/compare_matlab_python.py --input "C:\My Documents\data.tif"
   
   # Bad
   python scripts/compare_matlab_python.py --input C:\My Documents\data.tif
   ```

2. In batch files, use quotes:
   ```batch
   set INPUT_FILE="%~1"
   ```

3. In MATLAB, use single quotes:
   ```matlab
   run_matlab_vectorization('C:\My Documents\data.tif', 'output')
   ```

### Forward vs Backslash

**Symptom:** Path errors on Windows

**Solutions:**
1. Python accepts both `/` and `\`:
   ```python
   # Both work in Python
   path = "C:/Users/data/file.tif"
   path = "C:\\Users\\data\\file.tif"  # Note: double backslash
   ```

2. In batch files, use backslash:
   ```batch
   set PATH=C:\Users\data
   ```

3. Use Path object for cross-platform compatibility:
   ```python
   from pathlib import Path
   path = Path("data") / "file.tif"
   ```

### Relative vs Absolute Paths

**Symptom:** "File not found" when using relative paths

**Solutions:**
1. Use absolute paths when unsure:
   ```bash
   python scripts/compare_matlab_python.py \
       --input "C:\full\path\to\data.tif"
   ```

2. Convert relative to absolute:
   ```python
   from pathlib import Path
   abs_path = Path("relative/path").resolve()
   ```

3. Run commands from project root:
   ```bash
   cd /path/to/slavv2python
   python scripts/compare_matlab_python.py --input data/file.tif
   ```

---

## Output Issues

### Batch Folder Not Found

**Symptom:** Comparison script reports no MATLAB batch folder

**Solutions:**
1. Check MATLAB actually created output:
   ```bash
   ls comparison_output/matlab_results/
   # Should show batch_YYMMDD-HHmmss folder
   ```

2. Verify MATLAB completed successfully:
   - Check `matlab_run.log` for errors
   - Look for "Vectorization completed successfully" message

3. Check output directory permissions:
   ```bash
   # Ensure write permissions
   chmod -R u+w comparison_output  # Linux/Mac
   ```

### Empty or Corrupted .mat Files

**Symptom:** Cannot load MATLAB .mat files or files are empty

**Solutions:**
1. Check file size:
   ```bash
   ls -lh comparison_output/matlab_results/batch_*/vectors/*.mat
   ```

2. Try loading in MATLAB interactively:
   ```matlab
   load('comparison_output/matlab_results/batch_xxx/vectors/network_xxx.mat')
   whos  % List variables
   ```

3. Re-run MATLAB vectorization with fresh output directory

4. Check disk space:
   ```bash
   df -h  # Linux/Mac
   # Windows: Check drive properties
   ```

### Permission Denied

**Symptom:** Cannot write to output directory

**Solutions:**
1. Check directory permissions:
   ```bash
   ls -ld comparison_output
   ```

2. Run with appropriate permissions or change output location:
   ```bash
   python scripts/compare_matlab_python.py --output-dir ~/my_output
   ```

3. Close any programs that may have files open (MATLAB, Python IDEs)

---

## Comparison Issues

### Large Differences in Results

**Symptom:** Vertex/edge counts differ significantly between MATLAB and Python

**Possible Causes:**
- Parameter mismatch
- Different random seeds
- Numerical precision differences
- Algorithm version differences

**Solutions:**
1. Verify parameters match:
   ```bash
   # Check comparison_params.json matches defaults
   cat scripts/comparison_params.json
   ```

2. Check for parameter file usage:
   ```bash
   python scripts/compare_matlab_python.py \
       --params scripts/comparison_params.json \
       ...
   ```

3. Review MATLAB and Python source code versions:
   - Ensure legacy/Vectorization-Public is up to date
   - Check Python implementation commit

4. Run statistical analysis to quantify differences:
   ```bash
   python scripts/statistical_analysis.py \
       --comparison-report comparison_output/comparison_report.json
   ```

### Cannot Load MATLAB Results

**Symptom:** Warning "Could not load MATLAB output data"

**Solutions:**
1. Install scipy if missing:
   ```bash
   pip install scipy
   ```

2. Verify .mat file format compatibility:
   ```python
   from scipy.io import loadmat
   data = loadmat('path/to/file.mat', squeeze_me=True)
   print(data.keys())
   ```

3. Check for MATLAB version compatibility:
   - MATLAB R2019a+ uses MAT-File version 7.3
   - May need h5py for v7.3 files:
     ```bash
     pip install h5py
     ```

### Zero Matched Vertices

**Symptom:** Vertex matching reports 0 matched vertices

**Possible Causes:**
- Different coordinate systems
- Large position differences
- Empty vertex arrays

**Solutions:**
1. Check vertex data exists:
   ```python
   # In Python
   results['vertices']['positions'].shape  # Should be (N, 4)
   ```

2. Increase matching distance threshold (edit `compare_matlab_python.py`):
   ```python
   # In match_vertices function
   distance_threshold = 5.0  # Increase from 3.0
   ```

3. Verify coordinate systems match:
   - Check microns_per_voxel is consistent
   - Verify image was loaded correctly

---

## Performance Issues

### MATLAB Very Slow

**Symptom:** MATLAB taking hours to complete

**Solutions:**
1. Reduce image size for testing:
   ```python
   import tifffile
   import numpy as np
   
   # Load and downsample
   img = tifffile.imread('large.tif')
   small = img[::2, ::2, ::2]  # Half resolution
   tifffile.imwrite('small.tif', small)
   ```

2. Adjust processing parameters:
   - Reduce `scales_per_octave` (faster, less accurate)
   - Increase `radius_of_smallest_vessel` (fewer scales)
   - Reduce `max_voxels_per_node_energy`

3. Monitor MATLAB memory usage:
   ```matlab
   % In MATLAB
   memory  % Check memory usage
   ```

4. Close other applications to free memory

### Python Very Slow

**Symptom:** Python taking much longer than expected

**Solutions:**
1. Check if NumPy is using optimized BLAS:
   ```python
   import numpy as np
   np.show_config()  # Should show BLAS/LAPACK info
   ```

2. Install optimized NumPy:
   ```bash
   pip install numpy[mkl]  # Intel MKL optimization
   ```

3. Use memory mapping for large volumes:
   ```python
   volume = load_tiff_volume(path, memory_map=True)
   ```

4. Profile to find bottlenecks:
   ```python
   import cProfile
   cProfile.run('processor.process_image(...)')
   ```

---

## Debugging Tips

### Enable Verbose Logging

**Python:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**MATLAB:**
```matlab
% Add debug prints in run_matlab_vectorization.m
fprintf('DEBUG: At stage X\n');
```

### Test Components Separately

1. **Test MATLAB Only:**
   ```bash
   python scripts/compare_matlab_python.py \
       --skip-python \
       --input data/file.tif \
       --matlab-path "C:\...\matlab.exe" \
       --output-dir test_output
   ```

2. **Test Python Only:**
   ```bash
   python scripts/compare_matlab_python.py \
       --skip-matlab \
       --input data/file.tif \
       --matlab-path "dummy" \
       --output-dir test_output
   ```

3. **Test MATLAB Parser:**
   ```bash
   python scripts/matlab_output_parser.py comparison_output/matlab_results/batch_xxx
   ```

### Use Minimal Test Data

Create small test volume:
```python
import numpy as np
import tifffile

# Create 64x64x32 test volume
test_vol = np.random.randint(0, 255, (32, 64, 64), dtype=np.uint8)
# Add some bright spots (fake vessels)
test_vol[15:17, 30:34, 30:34] = 200
tifffile.imwrite('test_small.tif', test_vol)
```

### Check Intermediate Results

**MATLAB:**
```matlab
% Load intermediate results
load('comparison_output/matlab_results/batch_xxx/data/energy_xxx.mat')
whos
imagesc(max(energy, [], 3))  % Visualize energy
```

**Python:**
```python
import pickle
# Load checkpoint
with open('comparison_output/python_results/checkpoints/checkpoint_energy.pkl', 'rb') as f:
    data = pickle.load(f)
print(data.keys())
```

### Interactive MATLAB Debugging

Run MATLAB interactively for debugging:
```matlab
cd('C:\path\to\legacy\Vectorization-Public')
addpath('C:\path\to\slavv2python\scripts')

% Set parameters
input_file = 'C:\path\to\data.tif';
output_dir = 'C:\path\to\output';

% Run with debugging
dbstop if error
run_matlab_vectorization(input_file, output_dir)
```

### Compare with Known Good Results

If you have reference results:
```bash
# Run comparison against known good MATLAB output
python scripts/compare_matlab_python.py \
    --matlab-batch path/to/reference/batch_folder \
    --skip-matlab \
    ...
```

---

## Getting More Help

If issues persist:

1. **Check GitHub Issues:**
   - Search existing issues
   - Create new issue with:
     - Error messages (full traceback)
     - System info (OS, MATLAB version, Python version)
     - Minimal reproducible example

2. **Collect Debug Information:**
   ```bash
   # System info
   python --version
   pip list
   
   # MATLAB info
   matlab -batch "version; exit"
   
   # Run validation
   python scripts/validate_setup.py --matlab-path "..." > validation.txt
   ```

3. **Enable Full Logging:**
   ```bash
   python scripts/compare_matlab_python.py ... 2>&1 | tee full_log.txt
   ```

4. **Check Log Files:**
   - `comparison_output/matlab_results/matlab_run.log`
   - MATLAB command window output
   - Python traceback

---

## Quick Reference

### Essential Commands

```bash
# Validate setup
python scripts/validate_setup.py --matlab-path "C:\...\matlab.exe"

# Run full comparison
python scripts/compare_matlab_python.py \
    --input "data/file.tif" \
    --matlab-path "C:\...\matlab.exe" \
    --output-dir "comparison_output"

# Generate visualizations
python scripts/visualize_comparison.py \
    --comparison-report comparison_output/comparison_report.json \
    --output-dir comparison_output/visualizations

# Statistical analysis
python scripts/statistical_analysis.py \
    --comparison-report comparison_output/comparison_report.json \
    --output comparison_output/statistical_analysis.txt
```

### Common Error Patterns

| Error Message | Likely Cause | Quick Fix |
|---------------|--------------|-----------|
| "MATLAB executable not found" | Wrong path | Check MATLAB installation path |
| "ModuleNotFoundError: No module named 'scipy'" | Missing package | `pip install scipy` |
| "vectorize_V200 not found" | Wrong directory | Check legacy/Vectorization-Public exists |
| "No batch folders found" | MATLAB failed | Check matlab_run.log |
| "Could not load MATLAB data" | scipy not installed | `pip install scipy` |
| "Path contains spaces" | Unquoted path | Add quotes around path |
| "Permission denied" | File locked or no write access | Check permissions, close programs |

---

**Last Updated:** 2025-01-27
