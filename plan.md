# Plan: Using This Repo’s Tooling + Shared Utilities Well

This document is written to help humans and AI agents make high-quality changes in this repository with minimal churn.

## 0) Non-Negotiables (Repo Guardrails)

- Work from the repository root.
- Keep package code under `source/` and tests under `dev/tests/`.
- Prefer PowerShell-friendly commands on Windows.
- Use `logging` in library code (avoid `print()` except for CLI user-facing summaries).
- For MATLAB-parity-sensitive surfaces, keep diffs minimal and parity-focused.

## 1) Install the Right Dependency Set (Use `pyproject.toml` Extras)

This repo uses optional dependency groups. Install the smallest set that matches your task.

### Core (library-only)

```powershell
pip install -e .
```

### App work (Streamlit)

```powershell
pip install -e ".[app]"
```

### Development work (tests/lint/type)

```powershell
pip install -e ".[dev]"
```

### App + Dev (most common for feature work)

```powershell
pip install -e ".[app,dev]"
```

### Notes

- Avoid installing heavy extras unless needed (`ml`, `napari`, `cupy`, etc.).
- If you add a new dependency, prefer adding it to the narrowest extra that needs it.

## 2) Canonical Quality Gates (Use These Exactly)

These are the repo’s standard workflows.

### Format

```powershell
python -m ruff format source dev/tests
python -m ruff format --check source dev/tests
```

### Lint

```powershell
python -m ruff check source dev/tests
python -m ruff check source dev/tests --fix
```

### Type-check

```powershell
python -m mypy
```

### Tests

```powershell
python -m pytest dev/tests/
python -m pytest -m "unit or integration"
```

### Useful “broad sanity” check

```powershell
python -m compileall source dev/scripts
```

## 3) Test Placement Rules (So New Utilities Stay Maintainable)

Place tests by the owning package surface, not by the task that introduced them.

- Unit tests: `dev/tests/unit/<owner>/` where `<owner>` is one of `analysis`, `apps`, `core`, `io`, `runtime`, `utils`, `visualization`, `workflows`.
- Integration tests: `dev/tests/integration/`.
- UI tests: `dev/tests/ui/`.
- Diagnostic tests: `dev/tests/diagnostic/`.

Additional rules:

- Do not hand-add folder markers; `dev/tests/conftest.py` assigns markers by folder.
- Use the repo-local `tmp_path` fixture behavior (artifacts must stay under `dev/tmp_tests/`).

## 4) MATLAB Parity Safety (When It Applies)

If you touch parity/comparison/runtime logic (not typical for geometry refactors):

- Keep diffs minimal and parity-focused.
- Preserve staged run layout semantics.
- Run the parity diagnostic gate:

```powershell
python -m pytest dev/tests/diagnostic/test_comparison_setup.py
```

## 5) Shared Utilities Strategy (General)

### Goal

Create small, composable helpers that:

- have clear input/output contracts,
- are easy to test deterministically,
- reduce duplicated logic across modules,
- do not change algorithm behavior (especially on parity-sensitive paths).

### How to decide “utility vs inline code”

Extract a helper when at least one is true:

- the same logic appears in 2+ places,
- the logic has tricky edge cases (NaNs, zero vectors, empty arrays, dtype issues),
- the logic is a stable concept (e.g., “safe normalization”, “angle between vectors”).

Avoid extracting when:

- it’s a one-off transformation,
- it would obscure a parity-critical algorithm step.

## 6) Geometry/Numeric Utilities: Use the Existing Hub

This repo already has a geometry utility hub:

- Public API: `source/analysis/geometry.py` (re-exports)
- Implementation modules: `source/analysis/_geometry/`

Prefer adding shared geometry helpers under `source/analysis/_geometry/` and re-exporting them via `source/analysis/geometry.py` only when they are part of the intended public surface.

### Current patterns worth standardizing

These patterns already exist in multiple places and should be centralized:

1) **Safe vector normalization**

- Common need: normalize rows of a `(N,3)` array, handling zero vectors.
- Example duplication: `evaluate_registration()` normalizes with “replace zeros with 1.0”.

2) **Angle between vectors (degrees)**

- Common need: `degrees(arccos(clip(dot/(norms))))`.
- Example duplication: `calculate_branching_angles()` computes this inline.

3) **Apply voxel scaling consistently**

- Common need: `positions * microns_per_voxel`.
- Centralizing avoids subtle inconsistencies (dtype, shape checks).

### Recommended micro-API (small, testable)

Create (or consolidate into) a small module such as `source/analysis/_geometry/vector_math.py` containing helpers like:

- `safe_normalize_rows(vectors: np.ndarray, *, eps: float = 0.0) -> np.ndarray`
- `angle_degrees(a: np.ndarray, b: np.ndarray) -> float`
- `scaled_positions(positions: np.ndarray, scale: list[float] | np.ndarray) -> np.ndarray`

Then refactor call sites to use these helpers.

### Refactor workflow (geometry)

1. Identify duplication with search (e.g., `np.linalg.norm`, `np.clip`, `np.degrees(np.arccos(...))`).
2. Extract the smallest helper that preserves behavior.
3. Add unit tests under `dev/tests/unit/analysis/`.
4. Replace call sites.
5. Run:

```powershell
python -m ruff check source dev/tests --fix
python -m ruff format source dev/tests
python -m pytest -m "unit or integration"
```

### Behavior preservation checklist

- Preserve dtype expectations (most geometry code uses `float`).
- Preserve zero-vector handling (don’t introduce NaNs).
- Preserve shape validation (raise `ValueError` where existing code does).
- Avoid changing numeric tolerances unless explicitly intended.

## 7) “AI Agent” Checklist (Do This Every Time)

When making changes, an AI agent should:

- Read `AGENTS.md` and follow the canonical commands.
- Keep changes minimal and scoped.
- Prefer adding utilities in existing hubs (`source/analysis/_geometry/` for geometry).
- Add tests in the correct owner folder.
- Run ruff + pytest for the smallest relevant scope.
- If touching parity/runtime surfaces, follow the parity-safe instructions and run the diagnostic gate.
