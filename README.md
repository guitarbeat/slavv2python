# SLAVV Python

> 🔴 **Active developer work:** [TODO.md](docs/TODO.md) owns tasks. [Exact Proof Findings](docs/reference/core/EXACT_PROOF_FINDINGS.md) owns live parity status.

Python implementation of SLAVV for 3D vascular network extraction from
microscopy volumes. The public workflow is paper-first: run the native Python
pipeline with the default `paper` profile, export an authoritative
`network.json`, and use the developer-only parity tooling separately when you
need exact MATLAB artifact proof.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[app,workspace]"
slavv info
slavv run -i data/slavv_test_volume.tif -o slavv_output --export csv json
slavv analyze -i slavv_output\network.json
slavv plot -i slavv_output\network.json -o plots.html
```

## Common Commands

### Pipeline Execution
```powershell
slavv run -i volume.tif -o slavv_output --run-dir workspace\runs\sample_a --export csv json
slavv run -i volume.tif -o slavv_output --profile matlab_compat --export json
slavv analyze -i slavv_output\network.json
slavv plot -i slavv_output\network.json -o plots.html
slavv-app
```

### Monitoring Long-Running Jobs
```powershell
# Monitor active jobs
slavv jobs list
slavv jobs history --run-dir workspace\runs\my_run

# Check daemon status
slavv jobs daemon status
```

### Quality & Testing
```powershell
python -m pytest -m "unit or integration"
python -m ruff check slavv_python tests
python -m ruff format slavv_python tests
python -m mypy
```

## Python API

```python
from slavv_python import SlavvPipeline, load_tiff_volume

image = load_tiff_volume("volume.tif")
pipeline = SlavvPipeline()
results = pipeline.run(image, {"pipeline_profile": "paper"})
```

## Public Workflow

- `paper` is the default CLI and Streamlit profile. It runs the maintained
  native Python TIFF-to-network pipeline with paper-style Hessian projection.
- `matlab_compat` remains available when you want the older MATLAB-shaped
  defaults without entering the developer parity harness.
- `network.json` is the authoritative versioned export consumed by
  `slavv analyze` and `slavv plot`.

## What Is In This Repo

> **New to the codebase?** See [FOLDER_PURPOSE_GUIDE.md](docs/reference/core/FOLDER_PURPOSE_GUIDE.md) for a detailed explanation of when to use `slavv_python/` vs `tests/` vs `scripts/` vs `workspace/`.

- `slavv_python/`: Production package code (installed via pip)
- `tests/`: Automated test suite (runs in CI)
- `scripts/`: Developer utility scripts (committed but not installed) — [README](scripts/README.md)
- `workspace/`: Local experiment data (gitignored, personal) — [README](workspace/README.md)
- `docs/`: Maintained reference docs plus archival investigation notes

## Documentation

### Core Guides
- [Agent and workflow guide](docs/AGENTS.md) ⭐ — AI agent instructions, domain glossary, workflows
- [Documentation Index](docs/README.md) — Complete documentation map and navigation
- [Developer dashboard (tasks)](docs/TODO.md) — Active tasks and checkboxes
- [Contributing](docs/CONTRIBUTING.md) — Development workflow and PR process

### Reference Documentation
- [Reference Docs Index](docs/reference/README.md)
- [Paper Profile Workflow](docs/reference/workflow/PAPER_PROFILE.md)
- [Python Naming Guide](docs/reference/workflow/PYTHON_NAMING_GUIDE.md)
- [Test Placement Guide](tests/README.md)

### MATLAB Parity Work
- [Exact Proof Findings](docs/reference/core/EXACT_PROOF_FINDINGS.md) ⭐ — Live parity status, active runs, blockers
- [Parity Pre-Gate](docs/reference/workflow/PARITY_PRE_GATE.md) — Three-tier testing workflow
- [Parity Certification Guide](docs/reference/workflow/PARITY_CERTIFICATION_GUIDE.md) — Full certification commands
- [Parity Job Monitoring](docs/reference/workflow/PARITY_JOB_MONITORING.md) — Automated tracking for long experiments
- [MATLAB Method Implementation Plan](docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md)
- [MATLAB Parity Mapping](docs/reference/core/MATLAB_PARITY_MAPPING.md)

### Additional Resources
- [v22 Pointer Corruption Archive](docs/investigations/v22-pointer-corruption/README.md)
- [CHANGELOG.md](docs/CHANGELOG.md)

## Parity Closure Fast Path

**For AI agents:** See [Documentation Index](docs/README.md#-parity-closure-fast-path) for the complete parity workflow guide.

**Quick reference for current MATLAB parity work:**

1. **[Exact Proof Findings](docs/reference/core/EXACT_PROOF_FINDINGS.md)** ⭐ — Live run status, blockers, cold-start protocol
2. **[Parity Pre-Gate](docs/reference/workflow/PARITY_PRE_GATE.md)** — Crop harness commands and three-tier workflow
3. **[Parity Certification Guide](docs/reference/workflow/PARITY_CERTIFICATION_GUIDE.md)** — Canonical certification workflow
4. **[Parity Job Monitoring](docs/reference/workflow/PARITY_JOB_MONITORING.md)** — `slavv jobs` commands and `--monitor` flag
5. Use `slavv monitor --run-dir <run_root>` to watch active runs (add `--once` for snapshot)
6. Keep tasks in [TODO.md](docs/TODO.md); live status goes in EXACT_PROOF_FINDINGS.md

## Notes

- Structured `run_dir` metadata is the supported resumable workflow.
- `slavv monitor --run-dir <run_root>` is the primary run operations console.
- `slavv-app` remains the browser app for processing, analysis, curation, and
  visualization; it is not the canonical overnight run watcher.
- Deprecated watcher scripts have been removed; use `slavv monitor --run-dir <run_root>`
  or `slavv monitor --run-dir <run_root> --once` for run operations.
- The public product goal is a complete native Python implementation of the
  paper workflow; exact MATLAB artifact parity is a separate developer proof
  track.
- The legacy rich parity and MATLAB comparison harness has been removed from the
  public CLI surface.
- A developer-only parity runner is available at
  `scripts/parity_experiment.py` for rerunning Python `edges` or
  `network` against reusable staged comparison roots and for exact artifact
  proof against preserved MATLAB vectors.
