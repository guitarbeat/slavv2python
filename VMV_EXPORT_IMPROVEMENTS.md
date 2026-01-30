# VMV/CASX Export and Organization Improvements

**Date:** January 29, 2026  
**Status:** ✅ Implemented

## Summary

Implemented comprehensive VMV and CASX file export functionality with improved output organization, making it easy to view vascular networks in 3D using Blender with VessMorphoVis plugin.

## Problem Statement

- No VMV or CASX files were being generated in comparison outputs
- Python had export functions but they weren't being called
- MATLAB wasn't configured to export VMV/CASX files
- Output directories were hard to navigate without an index
- Users couldn't view vascular networks in 3D visualization tools

## Solution Overview

### 1. Python VMV/CASX Export ✅

**Modified:** `scripts/compare_matlab_python.py`

Added automatic export of multiple formats after Python vectorization:

```python
# Export VMV and CASX formats for visualization
visualizer = SLAVVVisualizer()
visualizer.export_results(results, vmv_path, format='vmv')
visualizer.export_results(results, casx_path, format='casx')
visualizer.export_results(results, csv_base, format='csv')
visualizer.export_results(results, json_path, format='json')
```

**Exports created:**
- `python_results/network.vmv` - VMV format for VessMorphoVis/Blender
- `python_results/network.casx` - CASX vascular network format
- `python_results/network_vertices.csv` - Vertex data (CSV)
- `python_results/network_edges.csv` - Edge data (CSV)
- `python_results/network.json` - Complete results (JSON)

### 2. MATLAB VMV/CASX Export ✅

**Modified:** `scripts/run_matlab_vectorization.m`

Added `SpecialOutput` parameter to request VMV and CASX export:

```matlab
'SpecialOutput', {{ 'vmv', 'casx' }}, ...  % Export VMV and CASX for visualization
```

**Exports created:**
- `matlab_results/batch_*/vectors/network_*.vmv`
- `matlab_results/batch_*/vectors/network_*.casx`

### 3. Automatic Manifest Generation ✅

**Created:** `scripts/generate_comparison_manifest.py`

Generates comprehensive `MANIFEST.md` for each comparison run with:
- File inventory organized by type (VMV, CASX, CSV, JSON, PNG, etc.)
- Comparison summary (performance, vertices, edges)
- File sizes and locations
- Instructions for viewing VMV files in Blender
- Directory structure tree
- Quick commands for common tasks

**Example manifest content:**
```markdown
# SLAVV Comparison Run: 20260128_python_with_plots

**Generated:** 2026-01-29 20:01:14
**Total Size:** 273.99 KB

## Comparison Summary
### Performance
- **MATLAB:** 3772.0s
- **Python:** 523.3s
- **Speedup:** 7.21x (Python faster)

## File Inventory
### 3D Visualization Files
**VMV Files** (VessMorphoVis/Blender):
- `python_results/network.vmv` (1.2 MB)

## How to View Results
### Viewing in Blender with VessMorphoVis
1. Install Blender
2. Install VessMorphoVis plugin
...
```

### 4. Batch Manifest Generator ✅

**Created:** `scripts/generate_all_manifests.py`

Utility to retroactively generate manifests for existing comparison directories:

```bash
python scripts/generate_all_manifests.py
```

Scans `comparisons/` directory and generates `MANIFEST.md` for any comparison lacking one.

### 5. Updated Documentation ✅

**Modified:** `scripts/README.md`

Added comprehensive sections on:
- 3D visualization with VessMorphoVis
- What files get exported
- How to view VMV files in Blender (step-by-step)
- Output management and cleanup commands

**Key additions:**
- Link to Blender download
- Link to VessMorphoVis plugin
- Clear instructions for enabling and using the plugin
- File path examples for finding VMV files

### 6. Integrated Manifest Generation ✅

**Modified:** `scripts/compare_matlab_python.py`

Automatic manifest generation at end of comparison:

```python
# Generate manifest automatically
from generate_comparison_manifest import generate_manifest
manifest_file = output_dir / 'MANIFEST.md'
generate_manifest(output_dir, manifest_file)
```

Now every comparison run automatically creates:
1. `comparison_report.json` - Detailed metrics
2. `summary.txt` - Human-readable summary
3. `MANIFEST.md` - File inventory and viewing instructions

## File Structure Improvements

### Before
```
comparisons/20260128_python_with_plots/
├── comparison_report.json
├── summary.txt
├── python_results/
│   └── python_comparison_parameters.json
└── visualizations/
    ├── count_comparison.png
    ├── summary_dashboard.png
    └── timing_breakdown.png
```

### After (Future Runs)
```
comparisons/YYYYMMDD_HHMMSS_comparison/
├── MANIFEST.md                      # ✨ NEW: File inventory + instructions
├── summary.txt                      
├── comparison_report.json           
├── matlab_results/
│   └── batch_YYMMDD-HHMMSS/
│       ├── vectors/
│       │   ├── network_*.vmv        # ✨ NEW: MATLAB VMV export
│       │   └── network_*.casx       # ✨ NEW: MATLAB CASX export
│       └── data/
├── python_results/
│   ├── network.vmv                  # ✨ NEW: Python VMV export
│   ├── network.casx                 # ✨ NEW: Python CASX export
│   ├── network_vertices.csv         # ✨ NEW: CSV export
│   ├── network_edges.csv            # ✨ NEW: CSV export
│   ├── network.json                 # ✨ NEW: JSON export
│   └── python_comparison_parameters.json
└── visualizations/
    ├── *.png
    └── 3d_renders/                  # Future: rendered views
```

## Viewing Networks in 3D

### Prerequisites
1. **Blender** (free, open-source): https://www.blender.org/download/
2. **VessMorphoVis plugin**: https://github.com/BlueBrain/VessMorphoVis

### Steps to View

1. **Install Blender**
   ```bash
   # Download and install from blender.org
   ```

2. **Install VessMorphoVis**
   - Download plugin from GitHub
   - Follow installation instructions in plugin README

3. **Enable Plugin**
   - Open Blender
   - Edit > Preferences > Add-ons
   - Search "VessMorphoVis"
   - Enable checkbox

4. **Load VMV File**
   - Open VessMorphoVis panel (usually right sidebar)
   - Click "Load Morphology"
   - Browse to: `comparisons/YYYYMMDD_HHMMSS/python_results/network.vmv`

5. **Visualize**
   - Choose visualization mode (e.g., color by radius, by strand)
   - Adjust rendering settings
   - Click "Render"
   - Export image or animation as needed

## Quick Commands

### List All Comparisons
```bash
python scripts/list_comparisons.py
```

### Generate Manifests for Existing Runs
```bash
python scripts/generate_all_manifests.py
```

### View Specific Manifest
```bash
cat comparisons/20260128_python_with_plots/MANIFEST.md
```

### Find All VMV Files
```bash
# Windows PowerShell
Get-ChildItem comparisons -Recurse -Filter "*.vmv"

# Unix/Mac
find comparisons -name "*.vmv"
```

## Technical Details

### Export Implementation

**Python VMV Export:**
- Uses `SLAVVVisualizer._export_vmv()` from `src/slavv/visualization.py`
- Converts vertex positions, radii, and strand connections to VMV format
- Writes ASCII text file following VMV specification

**MATLAB VMV Export:**
- Uses built-in `vectorize_V200` special output functionality
- Triggered by `'SpecialOutput', {'vmv', 'casx'}`
- Exports to `batch_*/vectors/` directory

### File Formats

**VMV Format:**
- ASCII text file
- Header with metadata (microns per voxel, etc.)
- Sections for samples (nodes) and connectivity
- Radius and position for each node

**CASX Format:**
- XML-based format
- Vascular network representation
- Developed by Linninger group at UIC

### Encoding Fixes

- Fixed Unicode encoding issues in manifest generation (Windows CMD compatibility)
- All manifest files now use UTF-8 encoding explicitly
- Removed Unicode symbols (arrows, checkmarks) for ASCII compatibility

## Benefits

1. **3D Visualization** - View vascular networks in interactive 3D with Blender
2. **Better Organization** - Clear file inventory in every comparison directory  
3. **Easy Discovery** - MANIFEST.md tells you exactly what files exist and where
4. **Standardized Output** - Consistent structure across all comparison runs
5. **Multiple Formats** - VMV, CASX, CSV, JSON for different use cases
6. **Self-Documenting** - Each run includes complete instructions for viewing

## Next Steps (Optional Enhancements)

- [ ] Pre-render 3D views automatically (PNG snapshots from multiple angles)
- [ ] Add interactive 3D viewer (web-based, no Blender required)
- [ ] Generate animated rotations of network (MP4/GIF)
- [ ] Add network statistics overlay on renders
- [ ] Create comparison gallery (side-by-side MATLAB vs Python renders)

## Testing

To test the improvements, run a new comparison:

```bash
python scripts/compare_matlab_python.py \
    --input "data/slavv_test_volume.tif" \
    --matlab-path "C:\Program Files\MATLAB\R2019a\bin\matlab.exe"
```

Check for:
- ✅ `python_results/network.vmv` exists
- ✅ `python_results/network.casx` exists
- ✅ `MANIFEST.md` exists and is readable
- ✅ CSV and JSON exports exist
- ✅ MATLAB VMV/CASX exports (if MATLAB produces vertices)

## Related Files

**Modified:**
- `scripts/compare_matlab_python.py` - Added VMV/CASX export calls
- `scripts/run_matlab_vectorization.m` - Added SpecialOutput parameter
- `scripts/README.md` - Added visualization section

**Created:**
- `scripts/generate_comparison_manifest.py` - Manifest generator
- `scripts/generate_all_manifests.py` - Batch manifest generator
- `VMV_EXPORT_IMPROVEMENTS.md` - This document

**Existing (Used):**
- `src/slavv/visualization.py` - Contains VMV/CASX export functions
- `src/slavv/io_utils.py` - File I/O utilities

## References

- VessMorphoVis: https://github.com/BlueBrain/VessMorphoVis
- Blender: https://www.blender.org
- SLAVV Paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009451
