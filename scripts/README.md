# MATLAB-Python Comparison Scripts

This directory contains scripts to run and compare the MATLAB and Python implementations of SLAVV vectorization.

## Files


### Interactive Notebooks (Recommended)
- **`notebooks/0_Setup_and_Validation.ipynb`** - Validate system setup and dependencies 
- **`notebooks/1_Comparison_Dashboard.ipynb`** - Interactive dashboard for exploring comparison results
- **`notebooks/2_Statistical_Analysis.ipynb`** - Detailed statistical reporting
- **`notebooks/3_Data_Management.ipynb`** - Manage output data and checkpoints

### Core Scripts (CLI)
- **`compare_matlab_python.py`** - Main CLI tool to run and compare implementations
- **`run_matlab_vectorization.m`** - MATLAB runner

### Shared Library (`src/slavv/dev/`)
- **`validation.py`** - Environment validation logic
- **`management.py`** - Data management logic
- **`matlab_parser.py`** - MATLAB output parsing logic
- **`metrics.py`** - Comparison mathematics
- **`viz.py`** - Shared visualization functions

### Legacy Analysis Tools
- **`matlab_output_parser.py`** - Parser for MATLAB .mat output files (extracts vertices, edges, network stats)


### Testing and Debugging
- **`test_matlab_setup.m`** - Simple test script to verify MATLAB environment
- **`test_comparison_setup.py`** - Test Python environment and file paths
- **`TROUBLESHOOTING.md`** - Comprehensive troubleshooting guide for common issues

## Quick Start

### Step 1: Validate Setup (Recommended)

Before running the comparison, validate your setup:

```bash
python scripts/validate_setup.py \
    --matlab-path "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" \
    --test-data "data/slavv_test_volume.tif" \
    --minimal-matlab-test
```

### Step 2: Run Comparison

```bash
python scripts/compare_matlab_python.py \
    --input "data/slavv_test_volume.tif" \
    --matlab-path "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" \
    --output-dir "comparisons/$(date +%Y%m%d_%H%M%S)_my_test"
```

Results are saved in timestamped directories under `comparisons/` with:
- `MANIFEST.md` - **File inventory and viewing instructions**
- `comparison_report.json` - Full results
- `summary.txt` - Human-readable summary  
- **`matlab_results/*.vmv`** - **VMV files for 3D visualization**
- **`python_results/network.vmv`** - **Python VMV export**
- **`python_results/network.casx`** - **Python CASX export**
- `visualizations/*.png` - Auto-generated comparison plots

### Step 3: View Results

```bash
# List all past runs
python scripts/list_comparisons.py

# View specific run summary
python scripts/list_comparisons.py --show 20260128_python_with_plots

# Or just read the summary file
cat comparisons/20260128_python_with_plots/summary.txt
```

### Step 4: Generate/Update Visualizations

```bash
python scripts/visualize_comparison.py \
    --comparison-report comparisons/YOUR_RUN/comparison_report.json \
    --output-dir comparisons/YOUR_RUN/visualizations
```

### Step 5: Clean Up Disk Space

```bash
# Analyze disk usage
python scripts/cleanup_comparisons.py --analyze

# Remove checkpoint files (safe - regeneratable)
python scripts/cleanup_comparisons.py --remove-checkpoints --confirm

# Archive old runs
python scripts/cleanup_comparisons.py --archive-old "comparison_output*" --confirm
```

### Test Python Only

```bash
python scripts/compare_matlab_python.py \
    --input "data/slavv_test_volume.tif" \
    --matlab-path "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" \
    --skip-matlab \
    --output-dir "comparison_output"
```

### Test MATLAB Only

```bash
python scripts/compare_matlab_python.py \
    --input "data/slavv_test_volume.tif" \
    --matlab-path "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" \
    --skip-python \
    --output-dir "comparison_output"
```

### Run MATLAB Directly (for debugging)

```bash
scripts\run_matlab_cli.bat "data\slavv_test_volume.tif" "comparison_output\matlab_results" "C:\Program Files\MATLAB\R2019a\bin\matlab.exe"
```

## Output Structure

```
comparison_output/
├── matlab_results/
│   ├── batch_YYMMDD-HHmmss/     # MATLAB batch output folder
│   │   ├── vectors/              # Network vectors (.mat files)
│   │   ├── data/                 # Intermediate data files
│   │   ├── settings/             # Parameter settings
│   │   └── timings.json          # Stage timing information
│   └── matlab_run.log            # MATLAB execution log
├── python_results/
│   ├── checkpoints/              # Python checkpoint files
│   ├── python_comparison_*.json  # Exported results
│   └── python_comparison_*.pkl   # Exported data
├── visualizations/               # Comparison plots
│   ├── count_comparison.png      # Vertex/edge/strand counts
│   ├── radius_distributions.png  # Radius histograms
│   ├── timing_breakdown.png      # Performance comparison
│   └── summary_dashboard.png     # Comprehensive overview
├── comparison_report.json        # Detailed comparison metrics
└── statistical_analysis.txt      # Statistical test results
```

## Parameters

Default parameters are defined in `comparison_params.json` and match the Python implementation defaults:

- `microns_per_voxel`: [1.0, 1.0, 1.0]
- `radius_of_smallest_vessel_in_microns`: 1.5
- `radius_of_largest_vessel_in_microns`: 50.0
- `approximating_PSF`: true
- `excitation_wavelength_in_microns`: 1.3
- `scales_per_octave`: 1.5
- `gaussian_to_ideal_ratio`: 1.0
- `spherical_to_annular_ratio`: 1.0
- `max_voxels_per_node_energy`: 100000

You can override these by providing a custom JSON file with `--params`.

## Comparison Features

The enhanced comparison framework provides:

1. **Detailed Result Comparison**
   - Vertex position matching with nearest-neighbor algorithm
   - Radius correlation analysis (Pearson & Spearman)
   - Edge count and length comparisons
   - Network topology analysis

2. **Statistical Analysis**
   - Kolmogorov-Smirnov test for distribution similarity
   - T-tests for mean differences
   - Effect sizes (Cohen's d, Hedge's g)
   - Bootstrap confidence intervals

3. **Visualization**
   - Count comparison bar charts
   - Radius distribution histograms
   - Timing breakdown charts
   - Summary dashboards

4. **Performance Metrics**
   - Stage-by-stage timing breakdown
   - Memory usage (if available)
   - Speedup calculations

## Advanced Usage

### Custom Parameters

```bash
python scripts/compare_matlab_python.py \
    --input "data/custom.tif" \
    --matlab-path "C:\...\matlab.exe" \
    --params custom_params.json \
    --output-dir "custom_output"
```

### Test Individual Components

```bash
# Test MATLAB only
python scripts/compare_matlab_python.py --skip-python ...

# Test Python only
python scripts/compare_matlab_python.py --skip-matlab ...

# Parse existing MATLAB output
python scripts/matlab_output_parser.py comparison_output/matlab_results/batch_xxx
```

### Generate Reports

```bash
# Full analysis pipeline
python scripts/validate_setup.py ...
python scripts/compare_matlab_python.py ...
python scripts/visualize_comparison.py ...
python scripts/statistical_analysis.py ...
```

## 3D Visualization with VessMorphoVis

Both MATLAB and Python now automatically export **VMV and CASX** files for viewing vascular networks in 3D!

### What Gets Exported

After running a comparison, you'll find:
- **Python**: `python_results/network.vmv` and `network.casx`
- **MATLAB**: `matlab_results/batch_*/vectors/network_*.vmv` and `*.casx`

### Viewing in Blender

1. Install [Blender](https://www.blender.org/download/) and [VessMorphoVis plugin](https://github.com/BlueBrain/VessMorphoVis)
2. Enable VessMorphoVis in Blender > Edit > Preferences > Add-ons
3. Open VessMorphoVis panel, click "Load Morphology"
4. Browse to your `.vmv` file and render

See each comparison's `MANIFEST.md` for complete instructions.

## Output Management

### List and View Comparisons

```bash
# List all comparison runs with summaries
python scripts/list_comparisons.py

# View specific run details
cat comparisons/YYYYMMDD_HHMMSS/MANIFEST.md
cat comparisons/YYYYMMDD_HHMMSS/summary.txt
```

### Clean Up

```bash
# Analyze disk usage
python scripts/cleanup_comparisons.py --analyze

# Remove checkpoints to save space
python scripts/cleanup_comparisons.py --remove-checkpoints
```

## Troubleshooting

For common issues and solutions, see **`TROUBLESHOOTING.md`**.

Quick diagnostics:
- Run `validate_setup.py` to check configuration
- Check `comparison_output/matlab_results/matlab_run.log` for MATLAB errors
- Verify MATLAB can find `vectorize_V200.m`
- Ensure all Python dependencies are installed: `pip install -r requirements.txt`

## Testing

Unit tests are available in `tests/unit/`:
- `test_matlab_parser.py` - Tests for MATLAB output parsing
- `test_comparison_metrics.py` - Tests for comparison functions

Run tests with:
```bash
pytest tests/unit/test_matlab_parser.py -v
pytest tests/unit/test_comparison_metrics.py -v
```

## Notes

- MATLAB R2019a+ uses the `-batch` flag for non-interactive execution
- The MATLAB script uses `Presumptive=true` and `VertexCuration='auto'`, `EdgeCuration='auto'` to skip all prompts
- Both implementations use the same test data file for fair comparison
- The comparison script loads and parses MATLAB .mat files using scipy
- Position matching uses a KD-tree for efficient nearest-neighbor search
- All timing information is automatically exported to JSON files
