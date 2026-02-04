# MATLAB-Python Comparison Scripts

This directory contains scripts to run and compare the MATLAB and Python implementations of SLAVV vectorization.

## Files

### Interactive Notebooks (Preferred)
- **`0_Setup_and_Validation.ipynb`** - Validate system setup and dependencies 
- **`1_Run_Comparison.ipynb`** - Interactive runner for the comparison pipeline
- **`2_Comparison_Dashboard.ipynb`** - Interactive dashboard for exploring comparison results
- **`3_Statistical_Analysis.ipynb`** - Detailed statistical reporting
- **`4_Data_Management.ipynb`** - Manage output data and checkpoints
- **`5_Tutorial.ipynb`** - General tutorial and usage examples

### Core Scripts (CLI)
- **`compare_matlab_python.py`** - Main CLI tool to run and compare implementations
- **`run_matlab_vectorization.m`** - MATLAB runner

### Shared Library (`source/slavv/dev/`)
- **`validation.py`** - Environment validation logic
- **`management.py`** - Data management logic
- **`matlab_parser.py`** - MATLAB output parsing logic
- **`metrics.py`** - Comparison mathematics
- **`viz.py`** - Shared visualization functions

### Testing and Debugging
- **`test_comparison_setup.py`** - Test Python environment and file paths

## Quick Start

### Step 1: Validate Setup
Open and run **`scripts/0_Setup_and_Validation.ipynb`**. This will check:
- MATLAB installation and version
- Python dependencies
- Test data integrity
- Disk space

### Step 2: Run Comparison (CLI or Notebook)
The comparison is best run from the command line due to long execution times, or use the interactive notebook:

**Option A: Notebook (Recommended for first run)**
Open **`scripts/1_Run_Comparison.ipynb`**.

**Option B: CLI**
```bash
python scripts/compare_matlab_python.py \
    --input "data/slavv_test_volume.tif" \
    --matlab-path "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" \
    --output-dir "comparisons/$(date +%Y%m%d_%H%M%S)_my_test"
```

### Step 3: Analyze Results (Notebooks)
Once the run is complete, use the notebooks for analysis:

1. **Dashboard**: Open `scripts/2_Comparison_Dashboard.ipynb` to interactively explore visualizations and metrics.
2. **Stats**: Open `scripts/3_Statistical_Analysis.ipynb` for rigorous statistical testing.
3. **Manage**: Open `scripts/4_Data_Management.ipynb` to list runs, generate manifests, and clean up disk space.

## Output Structure

```
comparison_output/
├── matlab_results/
│   ├── batch_YYMMDD-HHmmss/     # MATLAB batch output folder
│   │   ├── vectors/              # Network vectors (.mat files)
│   │   └── matlab_run.log        # MATLAB execution log
├── python_results/
│   ├── checkpoints/              # Python checkpoint files
│   ├── python_comparison_*.json  # Exported results
│   └── python_comparison_*.pkl   # Exported data
├── visualizations/               # Comparison plots (auto-generated)
├── comparison_report.json        # Detailed comparison metrics
├── MANIFEST.md                   # File inventory and viewing instructions
└── summary.txt                   # Human-readable summary
```

## Troubleshooting

### fast-check: Common Issues

| Issue | Quick Fix |
|-------|-----------|
| "MATLAB executable not found" | Check path in `0_Setup_and_Validation.ipynb` |
| "ModuleNotFoundError" | `pip install -r requirements.txt` |
| "vectorize_V200 not found" | Check `external/Vectorization-Public` exists |
| "Permission denied" | Close files/folders open in other apps |

### Detailed Solutions

#### MATLAB Issues

**MATLAB Not Found**
- Verify path in `notebooks/0_Setup_and_Validation.ipynb`.
- Ensure you point to the executable (e.g., `.../bin/matlab.exe`), not just the directory.

**`-batch` Flag Not Supported**
- MATLAB R2019a+ is required for the `-batch` flag.
- For older versions, edit `scripts/run_matlab_cli.bat` to use `-r` instead.

**MATLAB Hangs**
- Check available RAM (need ~8GB+).
- Clear MATLAB temp files: `matlab -batch "delete(fullfile(prefdir, '*.mat')); exit"`.

#### Python Issues

**Import Errors**
- Ensure you are running from the repository root.
- Ensure your environment is active (`conda activate slavv-env`).

**TIFF Loading Errors**
- Verify input file exists and is a valid 3D grayscale TIFF.
- Try loading with `tifffile.imread('path.tif')` in a notebook to test.

#### Output Issues

**Empty Results / Zero Vertices**
- If MATLAB produces 0 vertices, verify `external/Vectorization-Public` is correctly set up.
- If Python produces 0 vertices, check if input image is empty or parameters are too strict (e.g., `radius_of_smallest_vessel`).

#### Comparison Issues

**Large Differences**
- Numerical differences are expected due to algorithm variations.
- Significant topological differences (e.g., specific branches missing) might indicate parameter mismatches. Check `comparison_params.json`.

**Visualizing 3D Files**
- Use **VessMorphoVis** in Blender to view `.vmv` files generated in `matlab_results/` and `python_results/`.
- Refer to `MANIFEST.md` in any comparison output folder for specific viewing instructions.
