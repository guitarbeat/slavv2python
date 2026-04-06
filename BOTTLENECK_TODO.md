# MATLAB vs Python Comparison Bottlenecks TODO

## Completed in this pass

- [x] Add `--validate-only` mode to `workspace/scripts/cli/compare_matlab_python.py` and `orchestrate_comparison()` to run preflight checks without launching pipelines.
- [x] Add `--minimal-exports` mode to reduce Python comparison export overhead by skipping VMV/CASX/CSV/JSON extra exports.

## Next quick wins

- [ ] Add strict source selection for standalone comparison (`checkpoints-only` vs `json-only`) to avoid fallback probing cost.
- [ ] Add a `--resume-latest` helper that reuses the latest run root to reduce duplicate full runs.
- [ ] Print explicit "reuse this output dir to resume" guidance at the end of successful runs.

## Medium effort improvements

- [ ] Reduce duplicate filesystem scans by sharing one inventory pass for size + manifest + file inventory generation.
- [ ] Add a "deep compare" switch so full MATLAB parse (`load_matlab_batch_results`) is optional when only summary metrics are needed.
- [ ] Add a lightweight warm-up/health check command for MATLAB launch diagnostics.

## Reliability and ergonomics

- [ ] Improve output-root recommendation messaging when OneDrive and low-space conditions are detected.
- [ ] Add preflight caching for repeated checks in the same session and output root.
- [ ] Add docs examples for "fast parity loop" commands using `--minimal-exports` and `--validate-only`.
