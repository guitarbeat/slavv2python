# Adding Extraction Algorithms

This guide is the maintained contributor note for introducing a new extraction
algorithm to the Python SLAVV pipeline. It applies to new edge-generation
methods, new energy backends, and similar algorithmic additions that should be
reachable from the current `argparse` CLI and resumable pipeline.

## Design Rules

- Keep package code under `source/slavv/`.
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
| `source/slavv/utils/validation.py` | Validates new parameter values and sets defaults. |
| `source/slavv/apps/cli.py` | Exposes the new option on `slavv run`. |
| `source/slavv/core/pipeline.py` | Dispatches direct and resumable execution. |
| `source/slavv/core/*.py` | Holds the implementation itself. |
| `source/slavv/runtime/run_state.py` and stage manifests | Keep resumable artifacts inspectable if the new method adds files or optional tasks. |
| `dev/tests/unit/core/` and related owner-aligned tests | Locks behavior with deterministic coverage in direct and resumable modes. |

For edge extraction specifically, the maintained split today is:

- `source/slavv/core/edges.py` for stage orchestration and resumable helpers
- `source/slavv/core/edge_candidates.py` for candidate generation
- `source/slavv/core/edge_selection.py` for choice/cleanup logic

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
   messaging through the existing manifest/reporting surfaces.
5. Wire run metadata.
   If the algorithm adds durable artifacts or optional sub-steps, make sure
   run-state/stage-manifest surfaces can describe and rediscover them.
6. Add tests in the owner-aligned location.
   For core pipeline work, that usually means `dev/tests/unit/core/`.
7. Update docs.
   Add or refresh a focused reference note instead of leaving behavior only in
   code comments or TODO files.

## Parity And Layout Guardrails

When a change touches MATLAB parity or comparison-facing behavior, preserve the
staged run-root semantics (`01_Input/`, `02_Output/`, `03_Analysis/`,
`99_Metadata/`) and keep compatibility with legacy flat checkpoint/run layouts
where supported today.

Do not silently rename or relocate staged artifacts without adding matching
compatibility handling.

## Minimum Test Coverage

For a new algorithm mode, aim to add:

- validation coverage for the new parameter value
- one direct pipeline test
- one resumable-path test if the stage has resumable support
- regression coverage for any new diagnostics or persisted artifacts
- `python -m mypy` coverage if you widen the typed surface

Use the folder rules in `dev/tests/README.md` and the repo-local `tmp_path`
fixture from `dev/tests/conftest.py`.

## Validation Commands

Run the smallest set that matches your change scope, then expand to the
boundary-crossing gate when needed:

```powershell
python -m ruff check source tests
python -m mypy
python -m pytest -m "unit or integration"
```

If you changed parity/comparison behavior, also run:

```powershell
python -m pytest dev/tests/diagnostic/test_comparison_setup.py
```

## Documentation Checklist

Before considering the algorithm integrated, make sure all of these are true:

- the CLI help mentions the new mode
- the relevant reference doc explains when to use it
- any new staged artifacts are described in
  `docs/reference/COMPARISON_LAYOUT.md` if they become durable workflow
  outputs
- parity-sensitive changes cite the right chapter or report if they alter the
  MATLAB investigation surface

