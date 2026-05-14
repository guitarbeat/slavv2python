# SLAVV Developer Command Center

## Navigation

| Link | Purpose |
| :--- | :--- |
| [Live Proof Status](file:///d:/2P_Data/Aaron/slavv2python/docs/reference/core/EXACT_PROOF_FINDINGS.md) | Current v22 readouts and regression failures |
| [Investigation Findings](file:///d:/2P_Data/Aaron/slavv2python/docs/investigations/translation_pair_analysis/INVESTIGATION_FINDINGS.md) | Deep analysis of missing/extra translation pairs |
| [Policy & Implementation Phases](file:///d:/2P_Data/Aaron/slavv2python/docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md) | Canonical claim boundaries and long-term roadmap |
| [Parity Mapping](file:///d:/2P_Data/Aaron/slavv2python/docs/reference/core/MATLAB_PARITY_MAPPING.md) | Module-for-module structure against MATLAB |

---

## Current Velocity

- **Last Verified:** 930 / 1,197 matched pairs (77.7%) on a fresh exact run (generating 1,188 total candidates vs MATLAB's 1,197).
- **Measurement Source:** `validation_strel_fix_output_v8` experiment, 2026-05-12
- **Target:** 80% (Measures 2 & 3 — nearly achieved)
- **Active Status:** Active Development — Tightening filtering criteria
- **Last Updated:** 2026-05-12

> [!IMPORTANT]
> Re-run `capture-candidates` on a fresh clean checkpoint and `prove-exact` after any code change to update
> this metric. Do not compare against the contaminated `trace_order_fix` baseline.

---

## Task Dependencies

```
PARITY-002 (80% match)
├── Measure 2: Frontier insertion algorithm (in progress)
│   └── [RISK] PERF-001 frontier optimization may have introduced insertion-order bugs
├── Measure 3: Candidate filtering alignment (not started)
│   └── Depends on Measure 2 validation to avoid wasted gap analysis
├── Boundary conditions investigation (not started)
└── PARITY-003 (100% exact parity, downstream)

PAPER-001 (public workflow health)
└── Independent of parity work — tracks `slavv run` / export / app functionality
```

---

## Critical Priority

### [CRITICAL] PARITY-002: Achieve 80% Match Rate Milestone — IN PROGRESS

Close the gap from 670 to 80% matched MATLAB edge pairs on the native-first
exact route.

- **Proof Gate:** [EXACT_PROOF_FINDINGS.md](file:///d:/2P_Data/Aaron/slavv2python/docs/reference/core/EXACT_PROOF_FINDINGS.md)
- **Validate command:** `python scripts/cli/parity_experiment.py capture-candidates --source-run-root <clean_run> --oracle-root workspace/oracles/180709_E_batch_190910-103039 --dest-run-root <dest_run>`
- **Regression gate:** Improve the exact match count (currently 670) and close the gap on missing candidate pairs (currently 527 missing).

#### Measure 2: Frontier Insertion & Trace Generation — IN PROGRESS

Fix seed selection, insertion semantics, and frontier ordering for high-degree
vertices. We successfully debugged and implemented exact MATLAB behavior for
`_matlab_global_watershed_insert_available_location`.

- **Status:** Directional suppression bug fixed and candidate surplus eliminated! Candidate count reduced from 1,349 to 1,025, boosting matched pairs from 167 to 670 (56.0% match rate).
- [Target File](file:///d:/2P_Data/Aaron/slavv2python/slavv_python/core/edges/global_watershed.py)

Next steps:
- [x] Debug infinite-loop in `verify_watershed_fix.py` diagnostic run.
- [x] Implement and verify exact MATLAB splice behavior for frontier insertions.
- [x] Diagnose why Python generates significantly more candidate traces (1,349) compared to MATLAB (1,197) on exact oracle inputs.
- [x] Investigate candidate generation loop and safety pop conditions to identify false positive trace creation.
- [x] Analyze why 527 MATLAB traces are missing (target Top Missing Vertices: 1350, 92).
    - **Findings:** Identified a major misalignment in the "Universe Realignment Fix". Strel offsets were not rotated to match the (Z, X, Y) layout, causing distance penalties to be applied to wrong dimensions (e.g., Y-offset scaled by dz). Fixed in `common.py`.

#### Measure 3: Candidate Filtering Alignment — NOT STARTED

Tighten Python acceptance criteria to match MATLAB `get_edges_by_watershed`
filters. Blocked until Measure 2 is validated to avoid measuring against a
moving target.

- [Target File](file:///d:/2P_Data/Aaron/slavv2python/slavv_python/core/edges/candidate_generation.py)
- [Target File](file:///d:/2P_Data/Aaron/slavv2python/slavv_python/core/edges/cleanup.py)

Next steps:
- [ ] Wait for Measure 2 validation.
- [ ] Re-run gap analysis to identify remaining root causes.
- [ ] Align filtering order and thresholds with MATLAB.

#### Boundary Conditions — NOT STARTED

- [ ] Analyze discrepancies at volume edges after Measures 2 & 3 land.

#### Recent Completions

- [x] `trace_order_fix`: Seeded RNG for trace order shuffling (2026-05-05). Result: 149 → 404 matched pairs (+2.7x).
- [x] `parallel_verification`: Parallel processing run on D: drive (2026-05-05). Result: confirmed 33.8% baseline.

---

## High Priority

### [HIGH] PARITY-003: Achieve 100% Exact Parity

Complete mathematical alignment and unlock full developer promotion gate. All
stages (`vertices`, `edges`, `network`) must pass `prove-exact` on the
native-first exact route.

- **Effort:** XL (Continuous)
- **Status:** Blocked — waiting for PARITY-002 to close `edges.connections` gap.
- **Acceptance:** `prove-exact --stage all` passes.

---

### [HIGH] PAPER-001: Public Paper Workflow Health — NOT TRACKED YET

Ensure the public-facing `slavv run` workflow produces valid output without
MATLAB runtime dependencies. This is independent of exact parity work.

Next steps:
- [ ] Verify `slavv run -i volume.tif -o output --profile paper` completes end-to-end.
- [ ] Verify `network.json` export is well-formed and versioned.
- [ ] Verify `slavv analyze` and `slavv plot` consume `network.json` correctly.
- [ ] Verify `slavv-app` (Streamlit) launches and processes a sample dataset.
- [ ] Add integration test covering the paper-profile pipeline.

---

## Medium Priority

### [MEDIUM] PERF-001: Algorithm Performance Optimization — PAUSED

> [!WARNING]
> The `O(N²) → O(log N)` frontier optimization changed the data structure
> underlying the watershed algorithm. This may have introduced insertion-order
> bugs that Measure 2 is now chasing. Performance work on parity-sensitive
> surfaces is paused until PARITY-002 stabilizes.

- **Progress:**
  - [x] Optimized Global Watershed Frontier using native Priority Queue.
  - [x] Implemented Parallel Processing (`joblib`) across pipeline.
- [ ] Resume after PARITY-002 validates that frontier changes don't regress parity.

---

## Low Priority

### [LOW] INVEST-005: Execution Trace Comparison Framework — SUPERSEDED

> [!NOTE]
> The vertex-1350-specific investigation was superseded by the whole-frontier
> algorithmic fixes in Measure 2. The `trace-vertex` CLI subcommand and
> `compare_execution_traces.py` script remain available if per-vertex debugging
> is needed again.

Original scope was to isolate the first point of divergence for Hub Vertex 1350.
The frontier insertion rewrite in Measure 2 addresses the root cause at a higher
level.

- **Inspect Tool:** [compare_execution_traces.py](file:///d:/2P_Data/Aaron/slavv2python/scripts/cli/compare_execution_traces.py)
- **Trace Command:** `python scripts/cli/parity_experiment.py trace-vertex --source-run-root workspace/runs/seed --vertex-idx 1350 --output-trace trace.json`

---

**Maintainer:** Development Team
**Document Version:** 4.0 (Audit-driven rewrite)
