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

- **Active Priority:** **PAPER-001 (Public Paper Workflow Health)** — Fully Verified end-to-end (2026-05-20).
- **Parity Status:** 1062 / 1,197 matched pairs (88.7%) on a fresh exact run. **PAUSED** pending integration test suite expansion.
- **Measurement Source:** `validation_strel_fix_output_v29` experiment
- **Target:** 95% parity match rate (PARITY-002) after product workflow stabilization.
- **Last Updated:** 2026-05-20

> [!IMPORTANT]
> The parity track is paused. Ensure the native paper-profile pipeline and downstream CLI commands (`slavv run`, `slavv analyze`, `slavv plot`) remain fully operational before chasing the remaining 11.3% tie-breaking edge cases.

---

## Task Dependencies

```
PAPER-001 (public workflow health) 🔴 [CRITICAL]
└── Verified end-to-end, needs paper-profile integration tests

PARITY-002 (88.7% match) 🟡 [HIGH] — PAUSED
├── Measure 2: Frontier insertion algorithm (completed)
├── Measure 3: Candidate filtering alignment (not started)
│   └── Depends on post-stabilization gap analysis
└── PARITY-003 (100% exact parity, downstream)
```

---

## Critical Priority

### [CRITICAL] PAPER-001: Public Paper Workflow Health — ACTIVE PRIORITY

Ensure the public-facing `slavv run` workflow produces valid output without MATLAB runtime dependencies. The core product flow must remain verified end-to-end.

- **Proof Gate:** `TODO.md` / `PROJECT_STATUS.md`
- **Validate command:** `slavv run -i workspace/datasets/synthetic_volume.tif -o workspace/runs/test_run`
- **Downstream verify:** `slavv analyze -i workspace/runs/test_run/network.json` and `slavv plot -i workspace/runs/test_run/network.json`

Next steps:
- [x] Verify `slavv run -i volume.tif -o output --profile paper` completes end-to-end.
- [x] Verify `network.json` export is well-formed and versioned.
- [x] Verify `slavv analyze` and `slavv plot` consume `network.json` correctly.
- [x] Verify `slavv-app` (Streamlit) launches and processes a sample dataset.
- [ ] Add integration test covering the paper-profile pipeline in the CI/CD regression gate.

---

## High Priority

### [HIGH] PARITY-002: Achieve 95% Match Rate Milestone — PAUSED

Close the gap from 88.7% to 95% matched MATLAB edge pairs on the native-first exact route. Currently paused to focus resources on Priority 1 product health.

- **Proof Gate:** [EXACT_PROOF_FINDINGS.md](file:///d:/2P_Data/Aaron/slavv2python/docs/reference/core/EXACT_PROOF_FINDINGS.md)
- **Validate command:** `python scripts/cli/parity_experiment.py capture-candidates --source-run-root <clean_run> --oracle-root workspace/oracles/180709_E_batch_190910-103039 --dest-run-root <dest_run>`

#### Measure 3: Candidate Filtering Alignment — NOT STARTED
Tighten Python acceptance criteria to match MATLAB `get_edges_by_watershed` filters.
- [Target File](file:///d:/2P_Data/Aaron/slavv2python/slavv_python/processing/stages/edges/candidate_generation.py)
- [Target File](file:///d:/2P_Data/Aaron/slavv2python/slavv_python/processing/stages/edges/cleanup.py)

---

### [HIGH] PARITY-003: Achieve 100% Exact Parity — PAUSED

Complete mathematical alignment and unlock full developer promotion gate. All stages (`vertices`, `edges`, `network`) must pass `prove-exact`.

- **Status:** Paused — waiting for PARITY-002 to resume.

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
