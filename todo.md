# TODO

This checklist rolls up the current implementation plans into one execution
order:

- [docs/COMPARISON_OUTPUT_PREFLIGHT_IMPLEMENTATION_PLAN.md](docs/COMPARISON_OUTPUT_PREFLIGHT_IMPLEMENTATION_PLAN.md)
- [docs/COMPARISON_RESUME_TRANSPARENCY_IMPLEMENTATION_PLAN.md](docs/COMPARISON_RESUME_TRANSPARENCY_IMPLEMENTATION_PLAN.md)
- [docs/EDGE_PARITY_IMPLEMENTATION_PLAN.md](docs/EDGE_PARITY_IMPLEMENTATION_PLAN.md)

Recommended order:

1. Add output-root preflight so live MATLAB runs fail early when the run root is
   unsafe.
2. Add resume transparency so reruns are predictable and inspectable.
3. Tackle edge parity once the operational workflow is safer and easier to
   debug.

## Cross-Cutting Setup

- [x] Confirm the first-pass scope is comparison-mode first, with standalone
  MATLAB support only where it is cheap and low-risk.
- [ ] Define the shared metadata contract for new run artifacts and snapshot
  fields before implementing multiple features in parallel.
- [x] Decide which findings are fatal vs warning-only for preflight and resume
  status.
- [ ] Add or refresh reusable test fixtures for:
  - MATLAB `batch_*` directories
  - `matlab_resume_state.json`
  - `matlab_run.log`
  - run snapshots and manifests
- [ ] Document a canonical local output root for live MATLAB-enabled reruns so
  developers do not default back to risky paths.

## Track 1: Comparison Output Preflight

- [x] Create `source/slavv/evaluation/preflight.py`.
- [x] Refactor shared validation logic out of
  `source/slavv/evaluation/setup_checks.py` where it makes sense.
- [x] Implement a normalized preflight report for the selected output root.
- [x] Check:
  - path resolution
  - parent-directory creation
  - writability
  - free space
  - OneDrive-suspected paths
  - obviously non-local or otherwise risky roots when detectable
- [x] Call preflight from
  `source/slavv/evaluation/comparison.py` before MATLAB launch.
- [x] Block fatal launches before `run_matlab_vectorization(...)` starts.
- [x] Persist the result to `99_Metadata/output_preflight.json`.
- [x] Mirror the high-level outcome into the shared run snapshot.
- [x] Surface preflight warnings in the manifest and `slavv status`.
- [ ] Add minimal launcher-level safety checks in:
  - `workspace/scripts/cli/run_matlab_cli.bat`
  - `workspace/scripts/cli/run_matlab_cli.sh`
- [x] Add tests for:
  - healthy local output root
  - low-free-space root
  - unwritable root
  - OneDrive-suspected root
  - persisted preflight metadata on success and blocked launch

## Track 2: Comparison Resume Transparency

- [ ] Create `source/slavv/evaluation/matlab_status.py`.
- [ ] Implement normalized parsing for:
  - `matlab_resume_state.json`
  - selected `batch_*` folder completeness
  - partial stage artifacts
  - `matlab_run.log` tail
- [ ] Derive and persist rerun semantics fields such as:
  - batch folder being reused
  - resume mode
  - last completed stage
  - next stage
  - partial-artifact detection
  - rerun prediction
- [ ] Mirror MATLAB-specific state into `99_Metadata/run_snapshot.json`.
- [ ] Capture failure summaries and log tails on MATLAB failure.
- [ ] Distinguish:
  - fresh run
  - completed batch no-op
  - stage-boundary resume
  - mid-stage restart
  - stale running snapshot
- [ ] Update `slavv status` to show the MATLAB rerun decision clearly.
- [ ] Update the run manifest with:
  - `Resume Semantics`
  - `Authoritative Files`
  - `Failure Summary`
- [ ] Add tests for:
  - fresh run
  - completed batch no-op
  - stage resume from `vertices` to `edges`
  - mid-stage `energy` crash with partial artifacts
  - stale snapshot detection
  - imported MATLAB checkpoints causing Python rerun from `edges`

## Track 3: Edge Parity

- [ ] Expand parity diagnostics in:
  - `source/slavv/evaluation/metrics.py`
  - `source/slavv/core/tracing.py`
- [ ] Record per-origin candidate coverage so missing MATLAB endpoint pairs can
  be traced back to specific origin vertices.
- [ ] Measure separately:
  - raw frontier candidates
  - watershed supplement additions
  - cleanup rejections
  - missing MATLAB endpoint pairs
  - extra Python endpoint pairs
- [ ] Audit `_supplement_matlab_frontier_candidates_with_watershed_joins()`.
- [ ] Tighten supplement rules so they only add MATLAB-like joins.
- [ ] Compare `_trace_origin_edges_matlab_frontier()` against the MATLAB
  `get_edges_for_vertex.m` and `get_edges_by_watershed.m` behavior.
- [ ] Align frontier behavior around:
  - ordering
  - parent/child resolution
  - pruning
  - terminal hit handling
  - trace finalization
- [ ] Keep `_choose_edges_matlab_style()` focused on downstream dedupe/pruning
  rather than masking upstream semantic drift.
- [ ] Add or update parity-focused tests in:
  - `tests/unit/analysis/test_comparison_metrics.py`
  - `tests/unit/core/test_edge_cases.py`
  - `tests/integration/test_regression_edges.py`
- [ ] Re-run the diagnostic parity comparison and confirm:
  - missing MATLAB endpoint pairs decrease
  - extra Python candidate pairs decrease
  - final edge and strand counts converge to MATLAB

## Verification And Release Checklist

- [ ] Update docs after implementation:
  - `docs/COMPARISON_LAYOUT.md`
  - `docs/README.md`
  - any parity findings note that should reference the new workflow
- [ ] Run formatting:
  - `python -m ruff format source tests`
  - `python -m ruff check source tests`
- [ ] Run type checking:
  - `python -m mypy`
- [ ] Run targeted test coverage for modified modules first.
- [ ] Run broad regression coverage:
  - `python -m pytest -m "unit or integration"`
  - `python -m pytest tests/diagnostic/test_comparison_setup.py`
- [ ] Run `python -m compileall source workspace/scripts`.
- [ ] If MATLAB is available, run a fresh live comparison on a high-free-space
  local output root and verify:
  - preflight metadata is written
  - rerun semantics are visible
  - parity diagnostics are generated
  - final edge/strand status is easy to interpret

## Good First Slices

- [x] First slice for operational safety:
  implement output-root preflight plus persisted preflight metadata.
- [ ] First slice for transparency:
  implement normalized MATLAB status parsing plus rerun prediction in the run
  snapshot.
- [ ] First slice for parity:
  improve candidate coverage diagnostics before changing tracer behavior.
