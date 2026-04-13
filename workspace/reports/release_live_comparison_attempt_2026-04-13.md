# Release Live Comparison Attempt (2026-04-13)

## Scope

Started implementation of release verification from `todo.md` with Phase 1 setup gates and Phase 2 live comparison execution.

## Commands Executed

```powershell
# Diagnostic setup gate
python -m pytest tests/diagnostic/test_comparison_setup.py

# Output-root preflight
python workspace/scripts/cli/compare_matlab_python.py \
  --input workspace/tmp_debug_cli_case/input.tif \
  --matlab-path "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" \
  --output-dir C:\slavv_comparisons\release_verify_20260413 \
  --validate-only

# Baseline quality gate
python -m compileall source workspace/scripts
python -m ruff format --check source tests
python -m ruff check source tests
python -m mypy
python -m pytest -m "unit or integration"

# MATLAB health check
python workspace/scripts/cli/compare_matlab_python.py \
  --matlab-path "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" \
  --output-dir C:\slavv_comparisons\release_verify_20260413 \
  --matlab-health-check

# Live run retry with real TIFF fallback
python workspace/scripts/cli/compare_matlab_python.py \
  --input C:\Users\alw4834\Documents\slavv2python\.venv\Lib\site-packages\skimage\data\multipage.tif \
  --matlab-path "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" \
  --output-dir C:\slavv_comparisons\release_verify_20260413\live_20260413b
```

## Results

1. Phase 1 setup and baseline gates passed.
2. MATLAB health check passed in 43.0s.
3. Live run produced staged artifacts under `C:\slavv_comparisons\release_verify_20260413\live_20260413b`.
4. MATLAB failed in energy stage on tiny fallback TIFF (`multipage.tif`) with dimension mismatch in `energy_filter_V200`.
5. Python completed, but both MATLAB and Python produced zero vertices/edges/strands for this fallback input.

## Key Evidence

- Run manifest: `C:\slavv_comparisons\release_verify_20260413\live_20260413b\99_Metadata\run_manifest.md`
- Failure summary: `C:\slavv_comparisons\release_verify_20260413\live_20260413b\99_Metadata\matlab_failure_summary.json`
- MATLAB log: `C:\slavv_comparisons\release_verify_20260413\live_20260413b\01_Input\matlab_results\matlab_run.log`
- Comparison report: `C:\slavv_comparisons\release_verify_20260413\live_20260413b\03_Analysis\comparison_report.json`

## Performance Snapshot (from fallback run)

- MATLAB: 42.54s
- Python: 0.67s
- Speedup: 63.59x (Python faster)

Note: these timings are not release-grade because canonical data was not used.

## Blocker

Canonical input `data/slavv_test_volume.tif` is not present in this workspace, so the release live comparison step cannot be completed on canonical data yet.

## Next Command To Resume

Once canonical TIFF path is available, run:

```powershell
python workspace/scripts/cli/compare_matlab_python.py \
  --input <canonical_tiff_path> \
  --matlab-path "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" \
  --output-dir C:\slavv_comparisons\release_verify_20260413\live_canonical_20260413
```
