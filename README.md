# SLAVV Python

Python implementation of SLAVV for 3D vascular network extraction from
microscopy volumes.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[app,dev]"
slavv info
slavv run -i data/slavv_test_volume.tif -o slavv_output --export csv json
```

## Common Commands

```powershell
slavv run -i volume.tif -o slavv_output --run-dir dev\runs\sample_a --export csv json
slavv analyze -i slavv_output\network.json
slavv plot -i slavv_output\network.json -o plots.html
slavv-app
python -m pytest -m "unit or integration"
python -m ruff check source dev/tests
python -m ruff format source dev/tests
python -m mypy
```

## What Is In This Repo

- `source/`: pipeline, runtime, I/O, analysis, visualization, CLI, and app code
- `dev/tests/`: unit, integration, and UI coverage
- `dev/scripts/`: maintained helper scripts and benchmarks
- `docs/`: maintained reference docs plus archival investigation notes

## Documentation

- [AGENTS.md](AGENTS.md)
- [Documentation Index](docs/README.md)
- [Reference Docs](docs/reference/README.md)
- [MATLAB Method Implementation Plan](docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md)
- [MATLAB Parity Mapping](docs/reference/core/MATLAB_PARITY_MAPPING.md)
- [Exact Proof Findings](docs/reference/core/EXACT_PROOF_FINDINGS.md)
- [v22 Pointer Corruption Archive](docs/chapters/v22-pointer-corruption/README.md)
- [Test Placement Guide](dev/tests/README.md)
- [CHANGELOG.md](CHANGELOG.md)

## Notes

- Structured `run_dir` metadata is the supported resumable workflow.
- The legacy rich parity and MATLAB comparison harness has been removed from the
  public CLI surface.
- A developer-only parity runner is available at
  `dev/scripts/cli/parity_experiment.py` for rerunning Python `edges` or
  `network` against reusable staged comparison roots and for exact artifact
  proof against preserved MATLAB vectors.
