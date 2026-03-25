# SLAVV Python Port

Python reimplementation of SLAVV (Segmentation-Less, Automated, Vascular
Vectorization) for 3D vascular network extraction from microscopy volumes.
This repository ships the core library, a Streamlit UI, a command-line
interface, and MATLAB comparison helpers.

## Repository layout

| Path | Description |
| --- | --- |
| `source/slavv/` | Core package code, including processing, I/O, analysis, and visualization |
| `source/slavv/apps/` | User-facing entry points such as the CLI and Streamlit app |
| `tests/` | Unit, integration, UI, and diagnostic coverage |
| `workspace/scripts/cli/` | MATLAB comparison scripts and helper wrappers |
| `external/Vectorization-Public/` | Upstream MATLAB SLAVV checkout, when populated locally |
| `MATLAB_MAPPING.md` | Notes on MATLAB-to-Python mapping decisions |
| `EXPERIMENTS_REVIEW.md` | Review notes for experiment assets and repo state |
| `CHANGELOG.md` | Recent project changes and release-style notes |
| `pyproject.toml` | Package metadata and tool configuration |

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the package for the workflow you need:

```powershell
# Core library only
pip install -e .

# Streamlit app support
pip install -e ".[app]"

# Development tooling plus the app
pip install -e ".[app,dev]"
```

The `slavv-app` launcher requires the `app` extra because it depends on
Streamlit.

## Launching the app

After installing `.[app]`, start the UI with either of these commands:

```powershell
slavv-app
```

```powershell
python -m streamlit run source/slavv/apps/web_app.py
```

Open the local URL printed by Streamlit in your browser.

## CLI usage

Use the packaged CLI for headless or scripted workflows:

```powershell
slavv info
slavv run -i volume.tif -o slavv_output --export csv json
slavv import-matlab -b path\to\batch_260210-101213 -c my_checkpoints
```

## Programmatic usage

```python
from slavv import SLAVVProcessor

processor = SLAVVProcessor()
results = processor.process_image(image_data, params)

print(f"Vertices: {len(results['vertices']['positions'])}")
print(f"Edges: {len(results['edges']['traces'])}")
```

## MATLAB comparison helpers

Validate the local comparison surface first:

```powershell
pytest tests/diagnostic/test_comparison_setup.py
```

Run the comparison helper script from the repository root:

```powershell
python workspace/scripts/cli/compare_matlab_python.py `
    --input data/slavv_test_volume.tif `
    --matlab-path "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" `
    --output-dir comparison_output
```

## Testing and checks

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

## Contributing

- Keep package code under `source/slavv/`.
- Keep tests under `tests/` and use the existing `unit`, `integration`, `ui`,
  and `diagnostic` markers.
- Prefer type hints for new or modified public functions.
- Use `logging` in library code instead of `print()`.
- Preserve MATLAB parity where practical and add deterministic regression tests
  for behavior changes.

## License

This project is licensed under GNU GPL-3.0. See `LICENSE`.
