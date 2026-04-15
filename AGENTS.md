# AGENTS.md

Repository guidance for coding agents working in `slavv2python`.

## Scope

- Work from the repository root.
- Prefer PowerShell-friendly commands on Windows.
- Treat the commands in this file as the canonical workflows for the repo.
- Historical references to `biome` or `dev/examples/run_tutorial.py` are stale and do not apply to this project.

## Repository Map

- `source/slavv/`: core package code, including processing, I/O, analysis, visualization, and app entry points
- `dev/tests/`: unit, integration, UI, and diagnostic coverage
- `dev/scripts/cli/`: MATLAB comparison helpers and wrapper scripts
- `dev/scripts/maintenance/`: repo maintenance helpers for mapping and MATLAB script audits
- `dev/reports/`: legacy redirect location (reports are maintained under `docs/reports/`)
- `docs/`: maintained reference docs for MATLAB mapping, parity notes, report catalogs, and comparison run layout
- `external/Vectorization-Public/`: optional local checkout of the upstream MATLAB implementation

## Read First When Relevant

- `docs/README.md`: index for maintained reference docs and active chapter status.
- `dev/tests/README.md`: canonical test placement rules; new tests should mirror the owning package surface instead of the task name that introduced them.
- `dev/tests/conftest.py`: shared pytest behavior, including folder-based markers and the repo-local `tmp_path` fixture rooted under `dev/tmp_tests/`.
- `docs/reference/COMPARISON_LAYOUT.md`: canonical staged comparison layout details and naming conventions.
- `docs/reference/ADDING_EXTRACTION_ALGORITHMS.md`: contributor guide for adding new extraction algorithms.
- `source/slavv/parity/run_layout.py`: canonical staged comparison layout and legacy-path compatibility rules.
- `source/slavv/runtime/run_state.py`: structured run metadata, staged artifact locations, and legacy checkpoint compatibility.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependency set that matches the task:

```powershell
pip install -e .
pip install -e ".[app]"
pip install -e ".[app,dev]"
```

Install pre-commit hooks when working on repo changes:

```powershell
pre-commit install
```

## Canonical Commands

Format:

```powershell
python -m ruff format source tests
python -m ruff format --check source tests
```

Lint:

```powershell
python -m ruff check source tests
python -m ruff check source tests --fix
```

Type-check:

```powershell
python -m mypy
```

Notes:

- The current supported mypy gate is the repo-root `python -m mypy` command.
- It currently covers the CLI, Streamlit launcher, share-report, web app, run-state, and selected core pipeline modules.

Tests:

```powershell
python -m pytest dev/tests/
python -m pytest -m "unit or integration"
python -m pytest dev/tests/diagnostic/test_comparison_setup.py
```

Other useful checks:

```powershell
python -m compileall source dev/scripts
pre-commit run --all-files
```

## CLI And App Workflows

Package CLI:

```powershell
slavv info
slavv run -i volume.tif -o slavv_output --export csv json
slavv analyze -i slavv_output/network.json
slavv plot -i slavv_output/network.json -o plots.html
slavv import-matlab -b path\to\batch_260210-101213 -c my_checkpoints
```

Useful `slavv run` options:

```powershell
slavv run -i volume.tif -o slavv_output --run-dir dev\runs\sample_a
slavv run -i volume.tif -o slavv_output --checkpoint-dir checkpoints
slavv run -i volume.tif -o slavv_output --stop-after edges
slavv run -i volume.tif -o slavv_output --force-rerun-from vertices
```

Notes:

- `slavv run` writes structured run metadata under `<output>\_slavv_run` when `--run-dir` and `--checkpoint-dir` are both omitted.
- Prefer `--run-dir` for new resumable runs; `--checkpoint-dir` remains the legacy flat-checkpoint surface.
- `slavv analyze` can operate directly on the standard exported `network.json`.

Streamlit app:

```powershell
slavv-app
python -m streamlit run source/slavv/apps/web_app.py
```

The `slavv-app` launcher requires the `app` extra.
The ML curation flow accepts uploaded `.joblib` and `.pkl` model files directly.

## Recommended Workflows

### Small Code Change

1. Read the impacted module and its nearest tests first.
2. If you are adding or moving tests, verify placement against `dev/tests/README.md` and the marker behavior in `dev/tests/conftest.py`.
3. Run the smallest targeted pytest command that covers the change.
4. Run `python -m ruff check source tests --fix` and `python -m ruff format source tests` if you touched Python files.
5. Finish with `python -m pytest -m "unit or integration"` when the change crosses module boundaries.

### Regression Check

Use this before pushing substantial package changes:

```powershell
python -m compileall source dev/scripts
python -m ruff format --check source tests
python -m ruff check source tests
python -m mypy
python -m pytest -m "unit or integration"
```

If the change is UI-facing, also run the relevant `dev/tests/ui/` coverage. If the change touches diagnostics, comparison helpers, or environment setup, include the diagnostic tests as well.

### MATLAB Parity Workflow

Run this when touching MATLAB import, comparison, or parity-sensitive logic:

```powershell
python -m pytest dev/tests/diagnostic/test_comparison_setup.py
python dev/scripts/cli/compare_matlab_python.py `
    --input data/slavv_test_volume.tif `
    --matlab-path "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" `
    --output-dir comparison_output
```

Useful comparison flags:

```powershell
python dev/scripts/cli/compare_matlab_python.py --skip-matlab ...
python dev/scripts/cli/compare_matlab_python.py --skip-python ...
```

Parity layout conventions to preserve:

- Structured comparison runs use `01_Input/`, `02_Output/`, `03_Analysis/`, and `99_Metadata/` under the run root.
- Python comparison artifacts belong under `02_Output/python_results/`; run snapshots and normalized comparison params belong under `99_Metadata/`.
- Parity utilities should continue to accept both the staged layout and legacy flat layouts when existing code supports both.
- See `docs/reference/COMPARISON_LAYOUT.md` for the full layout reference and examples.

Expect the MATLAB workflow to require a populated `external/Vectorization-Public/` checkout and a valid local MATLAB installation.

## Repo-Specific Guardrails

- Keep package code under `source/slavv/`.
- Keep tests under `dev/tests/`; follow `dev/tests/README.md` for ownership-based placement and marker usage.
- Pytest markers are assigned by folder in `dev/tests/conftest.py`; files with `regression` in the node id also receive the `regression` marker.
- Use the repo-local `tmp_path` fixture behavior in `dev/tests/conftest.py` when writing tests; temporary test artifacts should stay under `dev/tmp_tests/`, not ad-hoc temp roots.
- Use `logging` in library code instead of `print()`. CLI commands may print user-facing summaries.
- Prefer `pathlib.Path` for filesystem-heavy code and use explicit text encodings such as `encoding="utf-8"` when writing repository-managed text artifacts.
- Prefer `from __future__ import annotations` in new Python modules to match the prevailing package style.
- Keep CLI surfaces aligned with the current `argparse`-based entrypoints in `source/slavv/apps/`; do not introduce a new CLI framework unless the task explicitly calls for it.
- Preserve the `source/` package layout and the existing console entrypoints declared in `pyproject.toml` (`slavv` and `slavv-app`).
- Preserve MATLAB parity where practical, and add deterministic regression tests for behavior changes.
- When touching parity or comparison code, preserve the staged run-root semantics implemented in `source/slavv/parity/run_layout.py` and `source/slavv/runtime/run_state.py`, including compatibility with legacy checkpoint directories where present.
- Prefer searching with `rg`, but exclude noisy generated trees like `dev/tmp_tests/` and vendored assets under `external/blender_resources/` unless the task explicitly targets them.
- Do not treat generated outputs under `comparisons/`, `comparison_output*/`, or cache directories as source inputs for code changes.


