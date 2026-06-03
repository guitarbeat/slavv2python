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

```powershell
slavv run -i volume.tif -o slavv_output --run-dir workspace\runs\sample_a --export csv json
slavv run -i volume.tif -o slavv_output --profile matlab_compat --export json
slavv analyze -i slavv_output\network.json
slavv plot -i slavv_output\network.json -o plots.html
slavv-app
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

- `slavv_python/`: engine, processing stages, analytics, storage, visualization, interface, and workflow code
- `tests/`: unit, integration, and UI coverage
- `scripts/`: maintained helper scripts and benchmarks
- `docs/`: maintained reference docs plus archival investigation notes

## Documentation

- [Agent and workflow guide](docs/AGENTS.md)
- [Developer dashboard (tasks)](docs/TODO.md)
- [Contributing](docs/CONTRIBUTING.md)
- [Documentation Index](docs/README.md)
- [Reference Docs](docs/reference/README.md)
- [Paper Profile Workflow](docs/reference/workflow/PAPER_PROFILE.md)
- [Python Naming Guide](docs/reference/workflow/PYTHON_NAMING_GUIDE.md)
- [MATLAB Method Implementation Plan](docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md)
- [MATLAB Parity Mapping](docs/reference/core/MATLAB_PARITY_MAPPING.md)
- [Exact Proof Findings](docs/reference/core/EXACT_PROOF_FINDINGS.md)
- [v22 Pointer Corruption Archive](docs/investigations/v22-pointer-corruption/README.md)
- [Test Placement Guide](tests/README.md)
- [CHANGELOG.md](docs/CHANGELOG.md)

## Parity Closure Fast Path

For the current MATLAB parity push:

1. Read [Exact Proof Findings](docs/reference/core/EXACT_PROOF_FINDINGS.md) for live truth.
2. Watch active structured runs with `slavv monitor --run-dir <run_root>`; use
   `--once` for a non-interactive snapshot.
3. Use [Parity Pre-Gate](docs/reference/workflow/PARITY_PRE_GATE.md) for crop proof commands.
4. Use [Parity Certification Guide](docs/reference/workflow/PARITY_CERTIFICATION_GUIDE.md) for canonical promotion and proof.
5. Keep tasks in [TODO.md](docs/TODO.md); do not duplicate live run status there.

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
  `scripts/cli/parity_experiment.py` for rerunning Python `edges` or
  `network` against reusable staged comparison roots and for exact artifact
  proof against preserved MATLAB vectors.
