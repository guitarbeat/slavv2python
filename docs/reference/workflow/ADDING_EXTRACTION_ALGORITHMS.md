# Adding Extraction Algorithms

[Up: Documentation Index](../../README.md)

This guide is the maintained contributor note for introducing a new extraction
algorithm to the Python SLAVV pipeline. It applies to new edge-generation
methods, new energy backends, and similar algorithmic additions that should be
reachable from the current `argparse` CLI and resumable pipeline.

## Design Rules

- Keep package code under `slavv_python/`.
- Preserve the existing `slavv` and `slavv-app` entrypoints in
  `pyproject.toml`.
- Extend the current parameter-validation and pipeline-dispatch surfaces rather
  than introducing a second algorithm registry or a new CLI framework.
- Prefer deterministic defaults. If an experimental mode is added, make the
  opt-in explicit in validated parameters and CLI flags.
- Preserve resumable behavior and inspectable run metadata. New algorithm
  artifacts should fit existing stage directories and manifest/reporting
  surfaces.

## Touch Points

Most new extraction algorithms need coordinated changes in these files:

| Surface | Why it matters |
| --- | --- |
| `slavv_python/utils/validation.py` | Validates new parameter values and sets defaults. |
| `slavv_python/apps/cli/parser.py` and `slavv_python/apps/cli/shared.py` | Expose the new option on `slavv run`. |
| `slavv_python/core/` and `slavv_python/workflows/` | Hold the implementation and the current pipeline orchestration. |
| `slavv_python/runtime/` and stage manifests | Keep resumable artifacts inspectable if the new method adds files or optional tasks. |
| `tests/unit/core/` and related owner-aligned tests | Lock behavior with deterministic coverage in direct and resumable modes. |

For edge extraction specifically, the maintained split today is:

- `slavv_python/core/edges/edges.py` for stage orchestration and resumable helpers
- `slavv_python/core/edges/candidate_generation.py` and related helpers for candidate generation
- `slavv_python/core/edges/selection.py` and `cleanup.py` for choice and cleanup logic

## Recommended Workflow

1. Add the parameter surface.
   Update validation defaults and CLI choices together so the new mode is
   impossible to request in one place and reject in another.
2. Implement the direct path.
   Keep the initial change as small and testable as possible.
3. Implement the resumable path.
   If the algorithm produces intermediate artifacts, persist them under the
   owning stage directory instead of ad-hoc scratch locations.
4. Expose diagnostics.
   Favor JSON or pickle artifacts under the stage directory and route summary
   messaging through the existing manifest and reporting surfaces.
5. Wire run metadata.
   If the algorithm adds durable artifacts or optional sub-steps, make sure
   run-state and stage-manifest surfaces can describe and rediscover them.
6. Add tests in the owner-aligned location.
   For core pipeline work, that usually means `tests/unit/core/`.
7. Update docs.
   Add or refresh a focused reference note instead of leaving behavior only in
   code comments or TODO files.

## Example: Adding An Energy Backend

Recent energy backends follow the existing `energy_method` surface instead of
introducing a second registry.

For example, the experimental `simpleitk_objectness` mode is integrated by:

- extending validation in `slavv_python/utils/validation.py`
- extending `slavv run --energy-method` choices in the CLI parser surfaces
- routing both direct and resumable execution through `slavv_python/core/energy/energy.py`
- keeping the default `hessian` path unchanged unless the new backend is
  explicitly selected
- documenting parameter differences when the backend cannot or should not
  emulate all controls used by the default production backend

The same pattern also applies to performance-oriented backends such as the
experimental `cupy_hessian` mode: keep the existing parameter and resumable
surfaces, document hardware and runtime requirements clearly, and avoid changing
the default CPU behavior unless the new backend is explicitly selected.

## Library Adoption Snapshot

The older external-library survey no longer needs to stand as a separate memo.
Its durable outcomes are now:

### Already adopted in the codebase

- `SimpleITK` as an optional exploratory energy backend in `slavv_python/core/energy/energy.py`
- `CuPy` as an optional experimental GPU energy backend in `slavv_python/core/energy/energy.py`
- `Zarr` for resumable energy storage and staged persistence
- `napari` as an optional curator surface in `slavv_python/visualization/napari_curator.py`
- `numba` as an optional acceleration path where the maintained code uses it

### Still-open future options

These are the only remaining library ideas still worth tracking at this level:

- `cuCIM` for GPU image operations with a scikit-image-like surface
- `Dask` for lazy chunked computation when volume size becomes the bottleneck
- `connected-components-3d` for faster 3D label cleanup helpers
- `MONAI` for any future learned segmentation or denoising branch
- `OpenCV` only if a targeted 2D preprocessing step becomes necessary

If one of these becomes active work, fold the decision into the relevant
maintained reference doc instead of creating a new broad survey note.

## Layout Guardrails

When a change touches resumable workflow behavior, preserve the structured
run-root semantics used by the current Python implementation.

Do not silently rename or relocate staged artifacts without updating the
runtime helpers, tests, and documentation together.

## Minimum Test Coverage

For a new algorithm mode, aim to add:

- validation coverage for the new parameter value
- one direct pipeline test
- one resumable-path test if the stage has resumable support
- regression coverage for any new diagnostics or persisted artifacts
- `python -m mypy` coverage if you widen the typed surface

Use the folder rules in `tests/README.md` and the repo-local `tmp_path`
fixture from `tests/conftest.py`.

## Validation Commands

Run the smallest set that matches your change scope, then expand to the
boundary-crossing gate when needed:

```powershell
python -m ruff check slavv_python tests
python -m mypy
python -m pytest -m "unit or integration"
```

## Documentation Checklist

Before considering the algorithm integrated, make sure all of these are true:

- the CLI help mentions the new mode
- the relevant reference doc explains when to use it
- any new durable staged artifacts are described in the relevant runtime or
  workflow reference docs
