# SLAVV Repository Guide & Instructions (AGENTS.md)

**For AI Agents:** This file is your primary navigation guide. Start with the [🧭 Work Decision Tree](#-work-decision-tree) below to find what to read.

Canonical instructions, domain glossary, and architecture guidelines for any AI coding agent working in the `slavv2python` repository. This file is automatically loaded into the AI's context.

**Quick Navigation:**
- **[🧭 Work Decision Tree](#-work-decision-tree)** ⭐ — **START HERE:** Find what to read based on your task
- **[Domain Glossary](#domain-glossary)** — All terminology definitions
- **[Repository Map](#repository-map)** — Directory structure
- **[Key Reference Documents](#key-reference-documents)** — Essential docs table
- **[Workflows](#workflows)** — Common development patterns

**Complete Table of Contents:**
- [Work Decision Tree](#-work-decision-tree) ← Start here to find what to read
- [Scope & Core Principles](#scope--core-principles)
- [Domain Glossary](#domain-glossary) ← All terminology definitions
- [Repository Map](#repository-map) ← Directory structure
- [Key Reference Documents](#key-reference-documents) ← Essential docs table
- [Setup & Installation](#setup--installation)
- [Quality Commands](#quality-commands)
- [CLI & Application Workflows](#cli--application-workflows)
- [Workflows](#workflows) ← Development patterns
- [Exact MATLAB Parity Rule](#exact-matlab-parity-rule)
- [Guardrails & Constraints](#guardrails--constraints)

> **Quick Navigation:**
> - [Work Decision Tree](#-work-decision-tree) — What to read based on your task
> - [Domain Glossary](#domain-glossary) — All domain terms
> - [Repository Map](#repository-map) — Directory structure
> - [Key Reference Documents](#key-reference-documents) — Essential docs
> - [Workflows](#workflows) — Common development patterns

---

## 🧭 Work Decision Tree

**Choose your path based on what you're working on:**

### I'm working on MATLAB parity
1. **Read [HANDOFF.md](.claude/HANDOFF.md)** — Current decision point and operating sequence
2. **Read [EXACT_PROOF_FINDINGS.md](docs/reference/core/EXACT_PROOF_FINDINGS.md)** — Live status, active runs, blockers
3. Check if a rerun is active: `slavv status-exact-run --run-dir <dir>` or check the `99_Metadata/parity_job.pid` file.
4. **Mandate**: All exact-route processing must use the **[Y, X, Z]** internal grid alignment with Fortran (F) memory order to match MATLAB's column-major tie-breaking.
5. Follow the operating sequence in [HANDOFF.md](.claude/HANDOFF.md) (findings cold-start defers there)
6. Use `--monitor` flag for long runs (see [PARITY_JOB_MONITORING.md](docs/reference/workflow/PARITY_JOB_MONITORING.md))
7. See [Parity Experiments](#parity-experiments) workflow below

### I'm fixing a bug or adding a feature
1. Read impacted module(s) and nearest tests
2. Check [PYTHON_NAMING_GUIDE.md](docs/reference/workflow/PYTHON_NAMING_GUIDE.md) for conventions
3. Verify test placement with [tests/README.md](tests/README.md)
4. Follow [Small Code Changes](#small-code-changes) workflow
5. If touching parity-sensitive code (energy, vertices, edges, network), also see parity workflow above

### I'm exploring the codebase
1. Start with [Repository Map](#repository-map) below
2. Review [TECHNICAL_ARCHITECTURE.md](docs/reference/core/TECHNICAL_ARCHITECTURE.md)
3. Check [Domain Glossary](#domain-glossary) for unfamiliar terms
4. See [Key Reference Documents](#key-reference-documents) table

### I'm setting up the environment
1. Follow [Setup & Installation](#setup--installation) below
2. Run [Quality Commands](#quality-commands) to verify setup
3. Try the [CLI workflows](#cli--application-workflows)

---

## Scope & Core Principles

- **Work Location:** Always work from the repository root directory.
- **Environment:** Prefer Windows PowerShell-friendly commands.
- **Source of Truth:** This file is the definitive guidance. When in doubt, defer here.

---

## Domain Glossary

> **Note:** This glossary is the **canonical source** for domain terminology. It is automatically loaded into AI agent context. The supplementary file [reference/core/GLOSSARY.md](docs/reference/core/GLOSSARY.md) is a browsable table view that may add technical detail.
> **Sync:** when adding or redefining a term, update both this section and `GLOSSARY.md`; if they ever disagree, this section wins.

### Lowest Linear Index Priority
The secondary tie-breaking rule for [Vertex](#vertex) and [Edge Discovery](#edge-discovery). When two voxels have identical energy values, the one with the lower Fortran-order linear index is prioritized.

### Paper Path
The baseline production [Pipeline](#pipeline) workflow. Optimized for multi-core scale-level parallelism, it utilizes `float32` precision and standard stride alignment. While fast on small volumes, it requires a significant memory footprint for the 4D scale stack.

### Exact Route (Innovation)
A memory-safe, bit-perfect [Pipeline](#pipeline) workflow that improves upon the original MATLAB architecture. It utilizes an **Incremental Octave-Chunked Engine** and `float64` precision to achieve [Certification](#certification) on massive volumes with minimal memory overhead.

### Pipeline
The authoritative sequence of computational stages (Energy → Vertices → Edges → Network) required to transform a 3D vascular volume into a vectorized graph representation.

### Vertex
A localized point of interest in the vascular volume, characterized by a 3D position, an estimated radius, and a local energy value.

### Seed Vertex
A [Vertex](#vertex) identified directly from the energy field as a local minimum. These serve as the initial discovery points for the [Pipeline](#pipeline).

### Bridge Vertex
A structural [Vertex](#vertex) inserted during edge selection to resolve overlaps or connectivity gaps. These are topologically necessary but were not originally identified as energy minima.

### Vertex Set
The authoritative collection of [Vertices](#vertex) for a given stage of a [Run](#run). A Vertex Set can contain both Seed and Bridge vertices.

### Edge Discovery
The process of identifying potential connectivity between [Vertices](#vertex) by analyzing the energy field.

### Tracing Discovery
An [Edge Discovery](#edge-discovery) strategy that identifies centerlines via frontier propagation from individual Seed Vertices.

### Watershed Discovery
An [Edge Discovery](#edge-discovery) strategy that partitions the volume into regional influence zones (catchment basins) to identify adjacent [Vertices](#vertex).

### Run State
The complete collection of data persisted during a [Run](#run).

### Stage Result
The authoritative output of a [Pipeline](#pipeline) stage, serving as the interface for subsequent stages.

### Checkpoint
Internal state persisted during a stage's execution to allow a [Run](#run) to recover from interruption or to skip recalculation.

### Artifact
Supplemental data produced by a stage for diagnostics, auditing, or visualization that is not strictly required for [Pipeline](#pipeline) progression.

### Oracle
Preserved MATLAB truth vectors and metadata for a specific dataset, stored under `workspace/oracles/`, used as the reference surface for exact parity comparison.

### Parity Run
A disposable developer execution under `workspace/runs/` that compares Python checkpoints against an [Oracle](#oracle) via the parity experiment harness.

### Parity Preflight
The memory, params-audit, and provenance checks run before a long [Parity Run](#parity-run) writer starts or resumes. Answers whether it is safe to launch, not whether Python matches MATLAB.
_Avoid_: Calling preflight a proof, or conflating it with `prove-exact`.

### Exact Proof Coordinator
The single orchestration surface that compares Python checkpoints against an [Oracle](#oracle) after they exist: `prove-exact`, candidate capture, LUT proof, and edge replay.
_Avoid_: Using "coordinator" for [Parity Preflight](#parity-preflight) or run-lifecycle concerns; those belong under `runs/preflight` and `resume-exact-run`.

### Certification
The state in which every required [Pipeline](#pipeline) stage passes its defined parity bar on a defined volume and workflow: Energy and Vertices use strict discrete equality plus `np.allclose` on continuous floats ([ADR 0011](docs/adr/0011-energy-float-certification-policy.md)); Edges and Network use the spatial bars in [ADR 0012](docs/adr/0012-edge-watershed-parity-bar.md) (ownership-map, trace tolerance, strand/bifurcation multisets)—not exact watershed edge-pair or strand-order equality.
_Avoid_: Treating strict-field `connections` / strand-count equality as the Phase 1 ship gate for Edges/Network; that is a separate [Strict-Field Stretch Goal](#strict-field-stretch-goal).

### Strict-Field Stretch Goal
An engineering target where Edges and Network also match MATLAB on strict discrete fields (`connections` counts, strand multisets with order-sensitive comparators)—tracked in [Exact Proof Findings](#exact-proof-findings) and iterated on the [Crop Harness Volume](#crop-harness-volume), but **not** a [Certification](#certification) blocker once ADR 0012 passes on the [Canonical Volume](#canonical-volume).
_Avoid_: Calling stretch progress "certified," blocking Phase 1 closure on strict-field gaps, or conflating it with the ADR 0012 ownership-map bar.

**Stretch KPIs:** Primary loop metric = **candidate generation overlap** (MATLAB final pairs present in Python candidates, crop harness). Milestone check = **strict-field** `prove-exact` / sequence delta on crop when overlap moves materially.
_Avoid_: Using ownership-map % as the stretch tracker (already at cert bar) or strict-field counts alone on every iteration.

**Stretch run root:** Use a **new** crop harness directory (e.g. `crop_M_exact_v3`) seeded via preflight from the prior crop run; rerun Edges only on current `main`. Do not treat stale pre-fix crop checkpoints as the stretch baseline.
_Avoid_: Logging stretch KPIs against `crop_M_exact` artifacts produced before watershed parity fixes landed on `main`.

**Phase 1 operating sequence:** (1) Refresh crop stretch run (`crop_M_exact_v3`, edges only) → log candidate-overlap KPI; (2) Launch canonical closure run (`canonical_full_v5`, edges → network) → per-stage ADR 0012 proof; (3) Declare [Phase 1 Closure](#phase-1-closure) if canonical passes; continue [Strict-Field Stretch Goal](#strict-field-stretch-goal) on crop without blocking ship.
_Avoid_: Starting the full-volume edges writer before the refreshed crop baseline, or running crop and canonical writers concurrently on memory-constrained hosts.

### Canonical Volume
The single full imaging volume chosen for a [Certification](#certification) milestone. Phase 1 exact-route canonical volume is full `180709_E`.
_Avoid_: Using a crop or Python-generated volume as the certification claim surface.

### Parity Pre-Gate
A faster developer loop that exercises the parity harness before [Certification](#certification) on the [Canonical Volume](#canonical-volume). Sequenced as: synthetic smoke, then real crop with its own [Oracle](#oracle), then canonical volume only for the final cert claim.

### Synthetic Fixture Volume
A Python-generated TIFF used for CI and harness smoke tests. Not paired with a preserved MATLAB [Oracle](#oracle) unless one is created explicitly for that volume.
_Avoid_: Calling this “synthetic certification” or reusing the `180709_E` oracle against it.

### Crop Harness Volume
A real subvolume cut from the `180709` imaging lineage, paired with its own promoted [Oracle](#oracle) produced from MATLAB vectorization on that same subvolume. Used for `prove-exact` iteration, not for the Phase 1 canonical cert claim unless explicitly promoted and documented as canonical.

### Phase 1 Specification
The single authoritative document for exact-route [Certification](#certification) on full `180709_E`: requirements and implementation together under `docs/plans/phase-1-exact-route-spec.md`.
_Avoid_: Maintaining separate brainstorm and plan files for the same initiative; use `docs/brainstorms/` only before the spec exists.

### Phase 1 Closure
The moment [Certification](#certification) is claimed for the exact route on full `180709_E`: per-stage `prove-exact --stage edges` and `--stage network` pass [ADR 0012](docs/adr/0012-edge-watershed-parity-bar.md) on the [Canonical Volume](#canonical-volume) run root against `180709_E_full_v2`, after Energy and Vertices already pass their bars. Does not require [Strict-Field Stretch Goal](#strict-field-stretch-goal) on the same run.
_Avoid_: Using `prove-exact-sequence` strict-field failure as the Phase 1 closure gate, or closing Phase 1 on [Crop Harness Volume](#crop-harness-volume) alone.

**Closure checkpoint policy:** Edges and Network checkpoints used for the closure proof must be produced by the **current** `main` code (rerun from edges on the canonical run root after any parity-sensitive merge), not frozen pre-fix artifacts.
_Avoid_: Claiming Phase 1 closure from stale edge/network checkpoints while stretch iteration uses fresher crop outputs.

**Closure run root:** Use a **new** canonical run directory (e.g. `canonical_full_v5`) seeded via preflight from the prior canonical run; carry forward certified Energy/Vertices, rerun Edges → Network only. Do not overwrite the prior canonical snapshot in place.
_Avoid_: `--force-rerun-from edges` on the historical claim run root when that run is the audit record for a prior milestone.

**Closure failure policy:** If the closure proof fails [ADR 0012](docs/adr/0012-edge-watershed-parity-bar.md) on the new canonical run, [Phase 1 Closure](#phase-1-closure) remains open. First triage measurement artifacts (checkpoint freshness, orientation/shape, oracle pairing, ownership-map probe)—do not assume a watershed code bug until those are ruled out. [Strict-Field Stretch Goal](#strict-field-stretch-goal) gaps do not reopen Phase 1 once ADR 0012 passes.
_Avoid_: Blocking Phase 1 on strict-field counts when ADR 0012 already passes, or chasing watershed fixes before ruling out stale checkpoints / orientation.

### Exact Proof Findings
The live status log for exact-parity work: active runs, `prove-exact` results, blockers, champion baselines, and a curated index of parity-related compound solutions under `docs/reference/core/EXACT_PROOF_FINDINGS.md`.
_Avoid_: Duplicating run status or solutions indexes in `TODO.md`; tasks stay in TODO, status and parity compound index stay in findings.

### Random Component Parity Suite
A seeded white-noise MATLAB R2019a/Python differential loop for fast Energy building-block checks (linspace, `interp3`, padded shape, valid flags). Structural fields gate CI; Hessian float ULP is advisory only. Not a [Certification](#certification) claim.
_Avoid_: Treating a green random-component run as crop or canonical `prove-exact` success.

---

## Repository Map

> **Confused about folder purposes?** See [docs/reference/core/FOLDER_PURPOSE_GUIDE.md](docs/reference/core/FOLDER_PURPOSE_GUIDE.md) for detailed explanations of when to use `slavv_python/` vs `tests/` vs `workspace/`.

**Four Top-Level Folders** (plus vendored `external/` third-party source):
- **`slavv_python/`** — Production package code (installed via pip)
- **`tests/`** — Automated test suite (runs in CI)
- **`workspace/`** — Local experiment artifacts (gitignored, personal)
- **`docs/`** — Maintained reference docs and archival investigation notes

```text
slavv2python/
├── slavv_python/                       # PRODUCTION PACKAGE (pip installable)
│   ├── engine/                         # Pipeline orchestration & lifecycle
│   │   └── state/                      # Run tracking, snapshots, resume
│   │       ├── run_ledger.py           # RunContext implementation
│   │       └── stage_handle.py         # StageController implementation
│   ├── pipeline/                       # Pipeline stages (Energy → Vertices → Edges → Network)
│   │   ├── energy/                     # EnergyManager, Hessian filtering, backends
│   │   ├── vertices/                   # Extraction, painting, selection
│   │   │   ├── manager.py              # Vertex lifecycle (run + run_resumable)
│   │   │   └── detection.py           # MATLAB-style candidate scan/choose
│   │   ├── edges/                      # Watershed, tracing, selection, cleanup
│   │   │   ├── discovery.py            # Edge discovery strategy seam
│   │   │   ├── manager.py              # Edge lifecycle (run + run_resumable)
│   │   │   └── matlab_get_edges_by_watershed.py, matlab_watershed_heap.py, ...  # Flat MATLAB-shaped edge modules
│   │   └── network/                    # Strand assembly, graph construction
│   │       └── manager.py              # Network lifecycle (run + run_resumable)
│   ├── image/                          # Image normalization, tiling
│   ├── analytics/                      # Analysis & metrics
│   │   ├── parity/                     # MATLAB exact proof harness (themed subpackages)
│   │   │   ├── commands.py              # slavv parity command registry (root)
│   │   │   ├── constants.py / utils.py  # shared parity constants + IO/hash helpers (root)
│   │   │   ├── proof/                   # coordinator, comparators, energy/ULP proofs, reports
│   │   │   ├── runs/                    # resume, jobs, monitor_daemon, preflight, writer_lease, bootstrap, execution
│   │   │   ├── oracle/                  # surfaces, oracle_artifacts, promotion, loaders, params_audit, models
│   │   │   ├── probes/                  # adaptive_probes, trace_comparator, crop_export, edge_artifacts, matlab_fail_fast
│   │   │   └── cli_handlers/            # cli_runs / cli_proofs / cli_diagnostics / cli_edges / cli_support
│   │   ├── curation/                   # Automated & ML curators
│   │   └── metrics/                    # Intensity, topology metrics
│   ├── storage/                        # Data I/O
│   │   ├── loaders/                    # TIFF, network loaders
│   │   └── exporters/                  # JSON v1 exporter
│   ├── interface/                      # User-facing surfaces
│   │   ├── cli/                        # argparse CLI
│   │   │   └── parity.py               # slavv parity <subcommand> entry point
│   │   ├── streamlit/                  # Streamlit web app
│   │   └── shared_services/            # Cross-UI service layer
│   ├── visualization/                  # Plotting & rendering
│   ├── workflows/                      # Pipeline orchestration helpers, profiles
│   ├── schema/                         # Data models
│   │   └── app_run.py                  # AppRunState (UI session envelope)
│   └── utils/                          # Validation, math, formatting
│
├── tests/                              # TEST SUITE (CI + local verification)
│   ├── unit/                           # By-owner unit tests (mirrors slavv_python/ structure)
│   │   ├── pipeline/                   # Tests for slavv_python/pipeline/
│   │   ├── analytics/                  # Tests for slavv_python/analytics/
│   │   ├── parity/                     # Tests for slavv_python/analytics/parity/
│   │   ├── engine/                     # Tests for slavv_python/engine/
│   │   ├── interface/                  # Tests for slavv_python/interface/
│   │   ├── storage/                    # Tests for slavv_python/storage/
│   │   ├── schema/                     # Tests for slavv_python/schema/
│   │   ├── utils/                      # Tests for slavv_python/utils/
│   │   ├── visualization/              # Tests for slavv_python/visualization/
│   │   └── workflows/                  # Tests for slavv_python/workflows/
│   ├── integration/                    # End-to-end & parity tests
│   ├── ui/                             # Streamlit & visualization tests
│   ├── runtime/                        # Run-state management tests
│   └── support/                        # Shared test builders & fixtures
│
├── docs/                               # Documentation
│   ├── reference/                      # Maintained technical references
│   └── investigations/                 # Archival investigation narratives
│
├── workspace/                          # LOCAL EXPERIMENT DATA (gitignored)
│   ├── oracles/                        # Preserved MATLAB oracle vectors
│   ├── runs/                           # Experiment trial runs
│   ├── reports/                        # Promoted proof summaries
│   ├── datasets/                       # Test datasets
│   └── scratch/                        # Temporary scratch files
│       └── matlab/                     # MATLAB driver scripts (local use only)
│
└── external/                           # Vendored dependencies
    └── Vectorization-Public/           # Canonical MATLAB source (submodule)
```

---

## Key Reference Documents

Read these first when working on relevant surfaces:

| Document | Path | Purpose |
|:---------|:-----|:--------|
| Developer Dashboard | [docs/TODO.md](docs/TODO.md) | Active tasks, planning hub (plans, brainstorms, solutions index) |
| Doc Index | [docs/README.md](docs/README.md) | Index for all maintained reference docs |
| Folder Purpose Guide | [docs/reference/core/FOLDER_PURPOSE_GUIDE.md](docs/reference/core/FOLDER_PURPOSE_GUIDE.md) | When to use `slavv_python/` vs `tests/` vs `workspace/` |
| MATLAB Parity Plan | [docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md](docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md) | Claim boundaries, source-of-truth hierarchy, remaining work |
| MATLAB-to-Python Map | [docs/reference/core/MATLAB_PARITY_MAPPING.md](docs/reference/core/MATLAB_PARITY_MAPPING.md) | Function-to-function mapping for exact parity |
| Exact Proof Findings | [docs/reference/core/EXACT_PROOF_FINDINGS.md](docs/reference/core/EXACT_PROOF_FINDINGS.md) | Live parity status, active blockers, proof results |
| Phase 1 handoff | [.claude/HANDOFF.md](.claude/HANDOFF.md) | Operator synthesis and closure run commands (`v3`/`v5`) |
| Parity Methodology | [docs/reference/core/PARITY_METHODOLOGY.md](docs/reference/core/PARITY_METHODOLOGY.md) | Literature-backed rationale for the tolerance-based parity bars (golden-master, allclose vs ULP, order-sensitivity, convention pitfalls); validates ADR 0011/0012 |
| Naming Guide | [docs/reference/workflow/PYTHON_NAMING_GUIDE.md](docs/reference/workflow/PYTHON_NAMING_GUIDE.md) | Python naming conventions and package surfaces |
| Testing Guide | [tests/README.md](tests/README.md) | Rules for test placement and markers |
| Parity Pre-Gate | [docs/reference/workflow/PARITY_PRE_GATE.md](docs/reference/workflow/PARITY_PRE_GATE.md) | Three-tier pre-gate (synthetic → crop → canonical) |
| Random Component Parity | [docs/reference/workflow/PARITY_RANDOM_COMPONENT_SUITE.md](docs/reference/workflow/PARITY_RANDOM_COMPONENT_SUITE.md) | Fast seeded noise differential ([ADR 0010](docs/adr/0010-random-component-parity-suite.md)) |
| Parity Certification | [docs/reference/workflow/PARITY_CERTIFICATION_GUIDE.md](docs/reference/workflow/PARITY_CERTIFICATION_GUIDE.md) | `prove-exact` / `prove-exact-sequence` on canonical volume |
| Documented Solutions | [docs/solutions/](docs/solutions/) | Searchable past fixes and workflows (`module`, `tags`, `problem_type` in YAML frontmatter); relevant when debugging parity, oracle promotion, or integration issues |
| ADRs | [docs/adr/](docs/adr/) | Architecture decisions (schema, executor, stage managers, parity coordinator) |
| Extraction Algorithms | [docs/reference/workflow/ADDING_EXTRACTION_ALGORITHMS.md](docs/reference/workflow/ADDING_EXTRACTION_ALGORITHMS.md) | Contributor guide for new algorithms |

---

## Setup & Installation

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependency set matching your task
pip install -e .                       # Core package only
pip install -e ".[app]"                # With Streamlit app dependencies
pip install -e ".[app,workspace]"      # Full developer environment (recommended)

# Install pre-commit hooks
pre-commit install
```

---

## Quality Commands

### Formatting & Linting
```powershell
python -m ruff format slavv_python tests
python -m ruff check slavv_python tests --fix
```

### Type Checking
```powershell
python -m mypy
```

### Running Tests
```powershell
python -m pytest tests/
python -m pytest -m "unit or integration"
```

### Full Regression Gate
Run before substantial changes:
```powershell
python -m compileall slavv_python scripts
python -m ruff format --check slavv_python tests
python -m ruff check slavv_python tests
python -m mypy
python -m pytest -m "unit or integration"
```

---

## CLI & Application Workflows

### Core CLI
```powershell
slavv info
slavv run -i volume.tif -o slavv_output --export csv json
slavv run -i volume.tif -o slavv_output --profile matlab_compat --export json
slavv analyze -i slavv_output/network.json
slavv plot -i slavv_output/network.json -o plots.html
```

### Advanced Options
```powershell
slavv run -i volume.tif -o slavv_output --run-dir workspace\runs\sample_a
slavv run -i volume.tif -o slavv_output --stop-after edges
slavv run -i volume.tif -o slavv_output --force-rerun-from vertices
```

> [!NOTE]
> `slavv run` writes structured run metadata under `<output>\_slavv_run` when `--run-dir` is omitted. The CLI defaults to the native `paper` profile.

### Run Operations Console
```powershell
slavv monitor --run-dir workspace\runs\oracle_180709_E\crop_M_exact
slavv monitor --run-dir workspace\runs\oracle_180709_E\crop_M_exact --once
slavv status --run-dir workspace\runs\oracle_180709_E\crop_M_exact
```

`slavv monitor` is the primary run-watching surface for structured pipeline and
parity runs. Streamlit remains the browser workflow for processing, analysis,
curation, and visualization; do not treat Streamlit as the canonical overnight
parity watcher.

**Checking a long run (e.g. canonical energy):** prefer the run's own signals
over tailing the harness log.
```powershell
# One-shot verdict (RUNNING/STALLED/COMPLETED/FAILED) + liveness + stage, run-dir only:
python scripts\check_parity_run.py --run-dir workspace\runs\oracle_180709_E\canonical_full_v3
# Live energy chunk rate + ETA (needs the run log with joblib "Done N tasks"):
python scripts\parity_run_throughput.py --run-dir <run> --log <run-log> --total-chunks <N>
```
Interpretation rules (learned the hard way):
- **Liveness = heartbeat *age*** (`resume_state.heartbeat_at` / `run_snapshot.updated_at`), **not** `started_at` — start-time clocks go stale across restarts.
- Under `n_jobs>1`, **run-dir progress is the merge cursor and lags the parallel compute**; the joblib `Done N tasks` log is the leading indicator. Never compute an ETA from the run-dir progress.
- A `(512,64,512)`-shaped energy checkpoint means an **orientation** problem, not values — confirm shape before interpreting a proof failure.

### Run Naming Convention
Long-running and background jobs are referred to by a memorable **food
codename** rather than the raw harness-assigned task ID, which is unreadable.
When launching such a job, assign a codename and announce it alongside the real
ID for traceability, then refer to it by the codename thereafter:

> 🥐 **Croissant** (`bfeprhlfo`) — canonical full-volume 180709_E certification.

The harness ID itself cannot be renamed; the codename is a maintained alias.
Prefer memorable-over-cryptic names for artifacts you *do* control as well —
run directories, oracles, and branches (e.g. `canonical_full_v2`, not a hash).

### Exact-Route Energy Parallelism
The exact-route Energy stage defaults to `n_jobs=1` (serial), which makes
full-volume certification a multi-day run. The chunk engine supports **threaded
parallelism that is provably bit-exact** (ordered merge, pure chunks, no RNG),
so pass `--n-jobs <N>` to `resume-exact-run` / `launch-exact-run` for a large
speedup (~5x/chunk at `n_jobs=6` on 8 cores). Changing `n_jobs` requires
`--force-rerun-from energy` (energy is not mid-stage resumable), and live
progress shows in the joblib `Done N tasks` log, not `resume_state.json`. See
[docs/solutions/parity/exact-energy-chunk-parallelism.md](docs/solutions/parity/exact-energy-chunk-parallelism.md).

### Streamlit App
```powershell
slavv-app
python -m streamlit run slavv_python/interface/streamlit/app.py
```

---

## Workflows

This section provides common development patterns. See [docs/README.md](docs/README.md) for the full documentation map.

### Small Code Changes
1. Read the impacted module and its nearest tests.
2. Verify test placement against `tests/README.md`.
3. Run the smallest targeted `pytest` command.
4. Format and lint: `ruff check --fix` and `ruff format`.
5. Run full suite if the change crosses module boundaries.

### Parity Experiments

**Prerequisites:** Read [docs/reference/core/EXACT_PROOF_FINDINGS.md](docs/reference/core/EXACT_PROOF_FINDINGS.md) BEFORE continuing parity work.

**Duplicate writer check:** If a crop rerun PID exists under `workspace/scratch/crop_energy_rerun_latest.pid`, check that process first. Use `slavv jobs list` to see active monitored jobs. Never start another writer on the same `crop_M_exact` run root while a writer is alive.

**Workflow:**

```powershell
# Promote oracle
slavv parity promote-oracle `
  --matlab-batch-dir D:\incoming\batch_260421-151654 `
  --oracle-root workspace\oracles\<oracle_id> `
  --dataset-file D:\datasets\volume.tif `
  --oracle-id <oracle_id>

# Run preflight check
slavv parity preflight-exact `
  --source-run-root workspace\runs\seed_run `
  --oracle-root workspace\oracles\<oracle_id> `
  --dest-run-root workspace\runs\my_current_code_trial

# Run full exact proof comparison
slavv parity prove-exact `
  --source-run-root workspace\runs\seed_run `
  --oracle-root workspace\oracles\<oracle_id> `
  --dest-run-root workspace\runs\my_current_code_trial `
  --stage all
```

## Exact MATLAB Parity Rule

For any MATLAB-parity-sensitive surface (especially `edges` and `network` stages):

- **Truth Source:** The MATLAB code under `external/Vectorization-Public/` is the canonical implementation.
- **Proof Gate:** Use `prove-exact` results and preserved MATLAB oracle vectors.
- **No Approximations:** Do not accept "close enough" replacements unless explicitly approved and documented.
- **1:1 Structure:** Python parity work must reproduce the same mathematical method and algorithm structure. Any undocumented deviation is a bug.

---

## Guardrails & Constraints

> [!IMPORTANT]
> **Max File Length:** Do not create or modify Python scripts to be more than **1000 lines** long.

| Rule | Detail |
|:-----|:-------|
| **Package Layout** | All package code under `slavv_python/`. Use surfaces from `PYTHON_NAMING_GUIDE.md`. |
| **Test Placement** | Tests under `tests/` per `tests/README.md`. Files with `regression` in the name get the `regression` marker automatically. |
| **Temporary Files** | Use the repo-local `tmp_path` fixture from `tests/conftest.py`. Test artifacts go in `workspace/scratch/tmp_tests/`. |
| **Logging** | Use `logging` in library code, not `print()`. CLI may print user-facing summaries. |
| **Path Handling** | Prefer `pathlib.Path`. Use explicit `encoding="utf-8"` for text files. |
| **Type Annotations** | Prefer `from __future__ import annotations` in all modules. |
| **CLI Framework** | Keep CLI under `slavv_python/interface/cli/` (argparse). No new CLI frameworks. |
| **Resumability** | Only the structured `run_dir` surface; no legacy checkpoint compatibility. |
| **Search Exclusions** | Exclude `workspace/scratch/tmp_tests/`, `external/blender_resources/`, and cache directories when searching. |
| **Precision** | **Mandatory float64**: All pipeline stages (Energy, Vertices, Edges) must use `float64` for coordinates, energies, and distance penalties to ensure bit-perfect MATLAB parity. |
| **Optimization** | **Bessel Sum Chunking**: Large kernel generations must use chunked summation (see `matlab_energy_filter_v200.py`) to prevent heap fragmentation OOM on 16GB systems. |
| **Scratch Files** | One-off scripts, logs, and experiment artifacts go in `workspace/scratch/`, not the repo root. |
| **Verify Before Commit** | Run the relevant tests — and the parity proof stage for parity-sensitive code (energy/vertices/edges/network) — and confirm green **before** committing. If on the default branch (`main`), create a feature branch first; commit/push only when asked. |
| **Line Counting** | PowerShell `Measure-Object -Line` silently drops blank lines and undercounts. Use Git Bash `wc -l` for file line counts, and re-verify a count before claiming a line-count change. |
| **Parity Metric** | Do not claim edge/network parity from raw **edge-pair overlap** — it can be inflated by coincidental wrong-grid matches. Use the spatial bars: voxel **ownership-map** agreement (edges) and **endpoint-pair / bifurcation multisets** + sub-voxel trace tolerance (network) per [ADR 0012](docs/adr/0012-edge-watershed-parity-bar.md). |
