# SLAVV Python Port

Python reimplementation of SLAVV (Segmentation-Less, Automated, Vascular
Vectorization) for 3D vascular network extraction from microscopy volumes.
This repository contains the package code, Streamlit app, CLI, MATLAB import
helpers, and parity/comparison tooling used to validate the port.

## Quick Start

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the package for the workflow you need:

```powershell
pip install -e .
pip install -e ".[app]"
pip install -e ".[app,dev]"
```

The `slavv-app` launcher requires the `app` extra because it depends on
Streamlit.

## Repository Layout

| Path | Description |
| --- | --- |
| `source/slavv/` | Core package code, including processing, I/O, analysis, visualization, and CLI/app entry points |
| `tests/` | Unit, integration, UI, regression, and diagnostic coverage |
| `workspace/scripts/` | MATLAB comparison wrappers and maintenance helpers |
| `workspace/reports/` | Archived tooling snapshots and other repo-local reference artifacts |
| `docs/` | Maintained reference docs for MATLAB mapping and comparison run layout |
| `external/Vectorization-Public/` | Upstream MATLAB SLAVV checkout, when populated locally |
| `CHANGELOG.md` | Recent project changes and development notes |
| `AGENTS.md` | Repository guidance for coding agents working in this repo |
| `pyproject.toml` | Package metadata and tool configuration |

## Common Commands

Use the packaged CLI for headless or scripted workflows:

```powershell
slavv info
slavv run -i volume.tif -o slavv_output --export csv json
slavv run -i volume.tif -o slavv_output --checkpoint-dir checkpoints
slavv import-matlab -b path\to\batch_260210-101213 -c my_checkpoints
```

The `slavv import-matlab` workflow imports staged MATLAB batches into
pipeline-compatible checkpoints, including curated vertex/edge artifacts and
the MATLAB HDF5 energy sidecar when present.

After installing `.[app]`, start the UI with either of these commands:

```powershell
slavv-app
python -m streamlit run source/slavv/apps/web_app.py
```

## Programmatic Usage

```python
from slavv import SLAVVProcessor

processor = SLAVVProcessor()
results = processor.process_image(image_data, params)

print(f"Vertices: {len(results['vertices']['positions'])}")
print(f"Edges: {len(results['edges']['traces'])}")
```

## MATLAB Import And Parity Workflows

Validate the local comparison surface first:

```powershell
python -m pytest tests/diagnostic/test_comparison_setup.py
```

Run the MATLAB/Python comparison helper from the repository root:

```powershell
python workspace/scripts/cli/compare_matlab_python.py `
    --input data/slavv_test_volume.tif `
    --matlab-path "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" `
    --output-dir comparison_output
```

Generated comparison runs should follow the staged layout documented in
[`docs/COMPARISON_LAYOUT.md`](docs/COMPARISON_LAYOUT.md): `01_Input`,
`02_Output`, `03_Analysis`, and `99_Metadata`.

As of March 26, 2026, exact vertex parity under MATLAB-energy control is in
place, and the parity-only edge cleanup plus strand-assembly path now uses
MATLAB-shaped ordering under `comparison_exact_network=True`. Live MATLAB
comparison runs remain the final confirmation surface for exact edge and strand
parity in a MATLAB-enabled environment. See
[`docs/MATLAB_MAPPING.md`](docs/MATLAB_MAPPING.md) for the current mapping and
parity status.

## Testing And Checks

Fast verification:

```powershell
python -m pytest -m "unit or integration"
```

Recommended local release checks:

```powershell
python -m compileall source workspace/scripts
python -m ruff format --check source tests
python -m ruff check source tests
python -m mypy
python -m pytest -m "unit or integration"
```

The current mypy gate is intentionally scoped to the packaged entry points.
Broader package-wide typing is still in progress, so new type-check coverage
should expand deliberately rather than claiming full-package enforcement.

## Documentation

Detailed docs that support the parity workflow now live under `docs/`:

| File | Purpose |
| --- | --- |
| [`docs/README.md`](docs/README.md) | Doc index and repository-reference entry point |
| [`docs/MATLAB_MAPPING.md`](docs/MATLAB_MAPPING.md) | High-level MATLAB-to-Python module mapping and current parity status |
| [`docs/COMPARISON_LAYOUT.md`](docs/COMPARISON_LAYOUT.md) | Canonical staged layout for MATLAB/Python comparison runs |

Repo-local maintenance helpers and archived tooling snapshots now live under
`workspace/scripts/maintenance/` and `workspace/reports/tooling/`.

## Contributing

- Keep package code under `source/slavv/`.
- Keep tests under `tests/` and use the existing `unit`, `integration`, `ui`,
  `diagnostic`, `slow`, and `regression` markers.
- Prefer type hints for new or modified public functions.
- Use `logging` in library code instead of `print()`.
- Preserve MATLAB parity where practical and add deterministic regression tests
  for behavior changes.

## License

This project is licensed under GNU GPL-3.0. See `LICENSE`.
