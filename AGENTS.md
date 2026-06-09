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
1. **Read [EXACT_PROOF_FINDINGS.md](docs/reference/core/EXACT_PROOF_FINDINGS.md) FIRST** — Live status, active runs, blockers
2. Check if a rerun is active: `slavv jobs list` or check `workspace/scratch/crop_energy_rerun_latest.pid`
3. Follow the cold-start protocol in EXACT_PROOF_FINDINGS.md
4. Review [PARITY_PRE_GATE.md](docs/reference/workflow/PARITY_PRE_GATE.md) for commands
5. Use `--monitor` flag for long runs (see [PARITY_JOB_MONITORING.md](docs/reference/workflow/PARITY_JOB_MONITORING.md))
6. See [Parity Experiments](#parity-experiments) workflow below

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

> **Note:** This glossary is the **canonical source** for domain terminology. It is automatically loaded into AI agent context. The supplementary file [reference/core/GLOSSARY.md](docs/reference/core/GLOSSARY.md) may contain additional technical details.

### Lowest Linear Index Priority
The secondary tie-breaking rule for [Vertex](#vertex) and [Edge Discovery](#edge-discovery). When two voxels have identical energy values, the one with the lower Fortran-order linear index is prioritized.

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

### Certification
The state in which sequential exact-parity gates report zero missing and zero extra for every required [Pipeline](#pipeline) stage on a defined volume and workflow.

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

### Exact Proof Findings
The live status log for exact-parity work: active runs, `prove-exact` results, blockers, champion baselines, and a curated index of parity-related compound solutions under `docs/reference/core/EXACT_PROOF_FINDINGS.md`.
_Avoid_: Duplicating run status or solutions indexes in `TODO.md`; tasks stay in TODO, status and parity compound index stay in findings.

---

## Repository Map

```text
slavv2python/
├── slavv_python/                       # Main package
│   ├── engine/                         # Pipeline orchestration & lifecycle
│   │   ├── context.py                  # Re-export barrel (RunContext, StageController)
│   │   └── state/                      # Run tracking, snapshots, resume
│   │       ├── run_ledger.py           # RunContext implementation
│   │       └── stage_handle.py         # StageController implementation
│   ├── processing/                     # Scientific computation
│   │   ├── image/                      # Normalization, tiling
│   │   └── stages/                     # Pipeline stages
│   │       ├── energy/                 # EnergyManager, Hessian filtering, backends
│   │       ├── vertices/               # Extraction, painting, selection
│   │       │   ├── manager.py          # Vertex lifecycle (run + run_resumable)
│   │       │   └── detection.py        # MATLAB-style candidate scan/choose
│   │       ├── edges/                  # Watershed, tracing, selection, cleanup
│   │       │   ├── discovery.py        # Edge discovery strategy seam
│   │       │   ├── manager.py          # Edge lifecycle (run + run_resumable)
│   │       │   └── matlab_algorithms/  # MATLAB-shaped parity shims
│   │       └── network/               # Strand assembly, graph construction
│   │           └── manager.py         # Network lifecycle (run + run_resumable)
│   ├── analytics/                      # Analysis & metrics
│   │   ├── parity/                     # MATLAB exact proof harness
│   │   │   ├── coordinator.py          # ExactProofCoordinator (prove / capture)
│   │   │   ├── preflight.py            # Memory gate + params audit before long runs
│   │   │   ├── resume.py               # resume-exact-run (clears stale running snapshot)
│   │   │   ├── bootstrap.py / surfaces.py / params_audit.py  # init-exact-run layout
│   │   │   └── counts.py               # Canonical RunCounts helpers
│   │   ├── curation/                   # Automated & ML curators
│   │   └── metrics/                    # Intensity, topology metrics
│   ├── storage/                        # Data I/O
│   │   ├── loaders/                    # TIFF, network loaders
│   │   └── exporters/                  # JSON v1 exporter
│   ├── interface/                      # User-facing surfaces
│   │   ├── cli/                        # argparse CLI
│   │   ├── streamlit/                  # Streamlit web app
│   │   └── shared_services/            # Cross-UI service layer
│   ├── visualization/                  # Plotting & rendering
│   ├── workflows/                      # Pipeline orchestration helpers, profiles
│   ├── schema/                         # Data models
│   │   └── app_run.py                  # AppRunState (UI session envelope)
│   └── utils/                          # Validation, math, formatting
│
├── tests/                              # Test suite
│   ├── unit/                           # By-owner unit tests
│   ├── integration/                    # End-to-end & parity tests
│   ├── ui/                             # Streamlit & visualization tests
│   ├── runtime/                        # Run-state management tests
│   └── support/                        # Shared test builders & fixtures
│
├── scripts/                            # Developer scripts
│   ├── cli/                            # Parity experiment harness
│   ├── matlab/                         # Headless MATLAB drivers (crop oracle vectorization)
│   └── diagnostics/                    # MATLAB artifact inspection
│
├── docs/                               # Documentation
│   ├── reference/                      # Maintained technical references
│   └── investigations/                 # Archival investigation narratives
│
├── workspace/                          # Developer experiment workspace
│   ├── oracles/                        # Preserved MATLAB oracle vectors
│   ├── runs/                           # Experiment trial runs
│   ├── reports/                        # Promoted proof summaries
│   ├── datasets/                       # Test datasets
│   └── scratch/                        # Temporary scratch files
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
| MATLAB Parity Plan | [docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md](docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md) | Claim boundaries, source-of-truth hierarchy, remaining work |
| MATLAB-to-Python Map | [docs/reference/core/MATLAB_PARITY_MAPPING.md](docs/reference/core/MATLAB_PARITY_MAPPING.md) | Function-to-function mapping for exact parity |
| Exact Proof Findings | [docs/reference/core/EXACT_PROOF_FINDINGS.md](docs/reference/core/EXACT_PROOF_FINDINGS.md) | Live parity status, active blockers, proof results, and cold-start protocol |
| Naming Guide | [docs/reference/workflow/PYTHON_NAMING_GUIDE.md](docs/reference/workflow/PYTHON_NAMING_GUIDE.md) | Python naming conventions and package surfaces |
| Testing Guide | [tests/README.md](tests/README.md) | Rules for test placement and markers |
| Parity Pre-Gate | [docs/reference/workflow/PARITY_PRE_GATE.md](docs/reference/workflow/PARITY_PRE_GATE.md) | Three-tier pre-gate (synthetic → crop → canonical) |
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
python scripts/cli/parity_experiment.py promote-oracle \
  --matlab-batch-dir D:\incoming\batch_260421-151654 \
  --oracle-root workspace\oracles\<oracle_id> \
  --dataset-file D:\datasets\volume.tif \
  --oracle-id <oracle_id>

# Run preflight check
python scripts/cli/parity_experiment.py preflight-exact \
  --source-run-root workspace\runs\seed_run \
  --oracle-root workspace\oracles\<oracle_id> \
  --dest-run-root workspace\runs\my_current_code_trial

# Run full exact proof comparison
python scripts/cli/parity_experiment.py prove-exact \
  --source-run-root workspace\runs\seed_run \
  --oracle-root workspace\oracles\<oracle_id> \
  --dest-run-root workspace\runs\my_current_code_trial \
  --stage all
```

---

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
| **Temporary Files** | Use the repo-local `tmp_path` fixture from `tests/conftest.py`. Test artifacts go in `tmp_tests/`. |
| **Logging** | Use `logging` in library code, not `print()`. CLI may print user-facing summaries. |
| **Path Handling** | Prefer `pathlib.Path`. Use explicit `encoding="utf-8"` for text files. |
| **Type Annotations** | Prefer `from __future__ import annotations` in all modules. |
| **CLI Framework** | Keep CLI under `slavv_python/interface/cli/` (argparse). No new CLI frameworks. |
| **Resumability** | Only the structured `run_dir` surface; no legacy checkpoint compatibility. |
| **Search Exclusions** | Exclude `tmp_tests/`, `external/blender_resources/`, and cache directories when searching. |
| **Scratch Files** | One-off scripts, logs, and experiment artifacts go in `workspace/scratch/`, not the repo root. |
