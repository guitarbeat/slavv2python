# MATLAB-Python Comparison Scripts

This directory contains scripts to run and compare the MATLAB and Python implementations of SLAVV vectorization.

## Files

### Core Scripts
- **`run_matlab_vectorization.m`** - Non-interactive MATLAB script wrapper for `vectorize_V200`
- **`run_matlab_cli.bat`** - Windows batch script to invoke MATLAB from command line
- **`compare_matlab_python.py`** - Main comparison script that runs both implementations with detailed analysis
- **`comparison_params.json`** - Shared parameter configuration file

### Analysis Tools
- **`matlab_output_parser.py`** - Parser for MATLAB .mat output files (extracts vertices, edges, network stats)
- **`visualize_comparison.py`** - Generate plots and charts comparing results
- **`statistical_analysis.py`** - Perform rigorous statistical tests on comparison results
- **`validate_setup.py`** - Pre-flight validation to check system configuration

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
- `comparison_report.json` - Full results
- `summary.txt` - Human-readable summary  
- Auto-generated visualizations

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
