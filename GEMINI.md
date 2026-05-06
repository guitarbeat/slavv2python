# GEMINI.md

Repository guidance for coding agents working in `slavv2python`.

## Scope

- Work from the repository root.
- Prefer PowerShell-friendly commands on Windows.
- Treat the commands in this file as the canonical workflows for the repo.

## Repository Map

- `source/`: core package code, including processing, I/O, analysis, visualization, and app entry points
- `dev/tests/`: unit, integration, UI, and diagnostic coverage
- `dev/scripts/`: maintained helper scripts and benchmarks
- `docs/`: maintained reference docs for the current Python codebase

## Read First When Relevant

- `docs/README.md`: index for maintained reference docs.
- `docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md`: canonical claim boundaries, source-of-truth hierarchy, and remaining work for fully implementing the released SLAVV method in Python.
- `docs/reference/core/MATLAB_PARITY_MAPPING.md`: canonical MATLAB-to-Python map for exact imported-MATLAB parity work.
- `docs/reference/workflow/PYTHON_NAMING_GUIDE.md`: preferred Python names, grouped package surfaces, and compatibility policy for the live codebase.
- `dev/tests/README.md`: canonical test placement rules; new tests should mirror the owning package surface instead of the task name that introduced them.
- `dev/tests/conftest.py`: shared pytest behavior, including folder-based markers and the repo-local `tmp_path` fixture rooted under `dev/tmp_tests/`.
- `docs/reference/workflow/ADDING_EXTRACTION_ALGORITHMS.md`: contributor guide for adding new extraction algorithms.
- `source/runtime/run_tracking/`: preferred runtime tracking package for structured run metadata and staged artifact locations.

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
python -m ruff format source dev/tests
python -m ruff format --check source dev/tests
```

Lint:

```powershell
python -m ruff check source dev/tests
python -m ruff check source dev/tests --fix
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
slavv run -i volume.tif -o slavv_output --profile matlab_compat --export json
slavv analyze -i slavv_output/network.json
slavv plot -i slavv_output/network.json -o plots.html
```

Useful `slavv run` options:

```powershell
slavv run -i volume.tif -o slavv_output --run-dir dev\runs\sample_a
slavv run -i volume.tif -o slavv_output --stop-after edges
slavv run -i volume.tif -o slavv_output --force-rerun-from vertices
```

Notes:

- `slavv run` writes structured run metadata under `<output>\_slavv_run` when `--run-dir` is omitted.
- The public CLI/app workflow defaults to the native `paper` profile.
- `slavv analyze` can operate directly on the authoritative exported `network.json`.

Streamlit app:

```powershell
slavv-app
python -m streamlit run source/apps/streamlit/app.py
```

The `slavv-app` launcher requires the `app` extra.
The ML curation flow accepts uploaded `.joblib` and `.pkl` model files directly.

## Recommended Workflows

### Small Code Change

1. Read the impacted module and its nearest tests first.
2. If you are adding or moving tests, verify placement against `dev/tests/README.md` and the marker behavior in `dev/tests/conftest.py`.
3. Run the smallest targeted pytest command that covers the change.
4. Run `python -m ruff check source dev/tests --fix` and `python -m ruff format source dev/tests` if you touched Python files.
5. Finish with `python -m pytest -m "unit or integration"` when the change crosses module boundaries.

### Regression Check

Use this before pushing substantial package changes:

```powershell
python -m compileall source dev/scripts
python -m ruff format --check source dev/tests
python -m ruff check source dev/tests
python -m mypy
python -m pytest -m "unit or integration"
```

If the change is UI-facing, also run the relevant `dev/tests/ui/` coverage.

### Developer Parity Experiment

Use this when you need a current-code rerun against a reusable staged comparison
root plus a preserved MATLAB oracle package:

```powershell
python dev/scripts/cli/parity_experiment.py promote-oracle `
    --matlab-batch-dir D:\incoming\batch_260421-151654 `
    --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a `
    --dataset-file D:\datasets\volume.tif `
    --oracle-id v22_a

python dev/scripts/cli/parity_experiment.py preflight-exact `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run `
    --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\my_current_code_trial

python dev/scripts/cli/parity_experiment.py prove-luts `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run `
    --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\my_current_code_trial

python dev/scripts/cli/parity_experiment.py capture-candidates `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run `
    --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\my_current_code_trial

python dev/scripts/cli/parity_experiment.py replay-edges `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run `
    --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\my_current_code_trial

python dev/scripts/cli/parity_experiment.py rerun-python `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\my_current_code_trial `
    --rerun-from edges

python dev/scripts/cli/parity_experiment.py summarize `
    --run-root D:\slavv_comparisons\experiments\live-parity\runs\my_current_code_trial

python dev/scripts/cli/parity_experiment.py prove-exact `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run `
    --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\my_current_code_trial `
    --stage all

python dev/scripts/cli/parity_experiment.py promote-report `
    --run-root D:\slavv_comparisons\experiments\live-parity\runs\my_current_code_trial
```

Notes:

- This runner is developer-only.
- The maintained experiment root splits `datasets/`, `oracles/`, `runs/`, and
  `reports/`, with `index.jsonl` at the root.
- `preflight-exact`, `prove-luts`, `capture-candidates`, and `replay-edges` are the fail-fast debugging funnel for the imported-MATLAB exact route.
- `rerun-python` and `summarize` provide count-level rerun summaries.
- `prove-exact` compares normalized Python checkpoints against preserved raw MATLAB vectors.
- `runs/` is disposable; use `promote-report` when a summary should be kept.
- `fail-fast` runs the cheap gates first and stops at the first failing gate before the full exact proof.
- It stays confined to the maintained developer parity runner surface.

## Exact MATLAB Parity Rule

For any MATLAB-parity-sensitive surface, especially the imported-MATLAB `edges` and `network`
stages, the required goal is exact method parity, not approximate behavioral similarity.

- Treat `docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md` as the canonical
  claim-boundary document for what counts as source-aligned, artifact-proven, or fully
  implemented in Python.
- Treat the MATLAB source under `external/Vectorization-Public/source/` as the canonical
  implementation.
- Treat `prove-exact` results and preserved MATLAB vectors as the proof gate.
- Treat the paper prose as explanatory context, not as a higher-priority spec than the released
  MATLAB code.
- Python parity work must reproduce the same mathematical method and algorithm structure 1:1 unless a
  deviation is explicitly documented and approved as non-parity work.
- Do not accept "close enough" replacements such as heuristic supplements, salvage passes, reordered
  ownership logic, or simplified local tracing when the MATLAB source uses a different global/shared
  method.
- Any undocumented deviation between Python and MATLAB on a parity surface should be treated as a bug,
  not as an implementation choice.
- When working on parity, audit the current Python path against the MATLAB source and the maintained
  mapping before making fixes.

## Repo-Specific Guardrails

- Keep package code under `source/`.
- Prefer the grouped package surfaces described in `docs/reference/workflow/PYTHON_NAMING_GUIDE.md` for new first-party imports and examples.
- Keep tests under `dev/tests/`; follow `dev/tests/README.md` for ownership-based placement and marker usage.
- Pytest markers are assigned by folder in `dev/tests/conftest.py`; files with `regression` in the node id also receive the `regression` marker.
- Use the repo-local `tmp_path` fixture behavior in `dev/tests/conftest.py` when writing tests; temporary test artifacts should stay under `dev/tmp_tests/`, not ad-hoc temp roots.
- Use `logging` in library code instead of `print()`. CLI commands may print user-facing summaries.
- Prefer `pathlib.Path` for filesystem-heavy code and use explicit text encodings such as `encoding="utf-8"` when writing repository-managed text artifacts.
- Prefer `from __future__ import annotations` in new Python modules to match the prevailing package style.
- Keep CLI surfaces aligned with the current `argparse`-based entrypoints in `source/apps/`; do not introduce a new CLI framework unless the task explicitly calls for it.
- Preserve the `source/` package layout and the existing console entrypoints declared in `pyproject.toml` (`slavv` and `slavv-app`).
- For MATLAB-parity work, preserve method parity with the upstream MATLAB source before optimizing,
  simplifying, or generalizing the Python implementation.
- Keep only the structured `run_dir` resumable surface; legacy checkpoint compatibility has been removed.
- Prefer searching with `rg`, but exclude noisy generated trees like `dev/tmp_tests/` and vendored assets under `external/blender_resources/` unless the task explicitly targets them.
- Do not treat generated outputs under `comparisons/`, `comparison_output*/`, or cache directories as source inputs for code changes.


