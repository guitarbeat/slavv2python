# TODO

This checklist rolls up the remaining implementation work and release
verification:

- [docs/chapters/imported-matlab-parity/EDGE_PARITY_IMPLEMENTATION_PLAN.md](docs/chapters/imported-matlab-parity/EDGE_PARITY_IMPLEMENTATION_PLAN.md)

Recommended order:

1. Add output-root preflight so live MATLAB runs fail early when the run root is
   unsafe.
2. Add resume transparency so reruns are predictable and inspectable.
3. Tackle edge parity once the operational workflow is safer and easier to
   debug.

## Completed (Consolidated)

- [x] Cross-cutting setup is complete:
  - comparison-mode first scope confirmed
  - shared metadata contract defined
  - fatal vs warning findings policy set
  - reusable MATLAB/run-state fixtures refreshed
  - canonical local output root documented
- [x] Track 1 comparison output preflight is complete:
  - preflight module, integration, persistence, manifest/status surfacing, and launcher checks implemented
  - preflight test matrix (healthy, low-space, unwritable, OneDrive-suspected, persisted metadata) added
- [x] Track 2 comparison resume transparency is complete:
  - normalized MATLAB status parsing, rerun semantics, snapshot/manifest updates, and failure summaries implemented
  - resume behavior test matrix (fresh/no-op/stage resume/mid-stage crash/stale snapshot/imported checkpoints) added
- [x] Track 3 edge parity work is complete:
  - diagnostics expanded and MATLAB frontier behavior alignment completed
  - parity-focused tests updated
  - diagnostic reruns confirmed reduced candidate/endpoint mismatch trends
- [x] Verification completed (non-MATLAB dependent):
  - docs updated
  - formatting/linting/type-checking completed
  - targeted and broad regression tests completed
  - `python -m compileall source workspace/scripts` completed
- [x] Initial milestone slices complete:
  - operational safety (preflight + metadata)
  - transparency (status parsing + rerun prediction)
  - parity diagnostics baseline improvements

## Remaining For Release

- [ ] If MATLAB is available, run a fresh live comparison on a high-free-space
  local output root and verify:
  - preflight metadata is written
  - rerun semantics are visible
  - parity diagnostics are generated
  - final edge/strand status is easy to interpret
- [ ] Final live comparison audit on canonical data.
- [ ] Snapshot performance metrics for native and parity paths.
- [ ] Prepare final parity report and findings summary.

### Implementation Progress (2026-04-13)

- [x] Phase 1 setup/baseline gates executed:
  - `pytest tests/diagnostic/test_comparison_setup.py` passed
  - output-root preflight validate-only run passed on `C:\slavv_comparisons\release_verify_20260413`
  - `compileall`, `ruff format --check`, `ruff check`, `mypy`, and `pytest -m "unit or integration"` passed
- [x] MATLAB health check executed:
  - `--matlab-health-check` passed (43.0s) on `C:\slavv_comparisons\release_verify_20260413`
- [ ] Fresh canonical live comparison remains open:
  - first attempt used a non-canonical fixture (`workspace/tmp_debug_cli_case/input.tif`) and failed because it is not a valid TIFF
  - second attempt used a real TIFF (`skimage/data/multipage.tif`) and produced staged artifacts, but MATLAB failed in energy stage on this tiny fallback input
  - failure evidence is captured in `C:\slavv_comparisons\release_verify_20260413\live_20260413b\99_Metadata\run_manifest.md` and `matlab_failure_summary.json`

## Additional Known Issues

- [x] Broaden entry-point type coverage across the CLI, web app, share-report, and run-state surfaces.
- [ ] Expand documentation for custom energy computation methods.
- [ ] Optimize peak memory usage during Hessian eigenvalue computation.
- [x] Document advanced `slavv analyze` metrics.
- [ ] Continue expanding typed coverage deeper into `analysis/` and other scientific modules.
- [ ] Add detailed contributor guide for adding new extraction algorithms.
