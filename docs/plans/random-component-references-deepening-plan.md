---
title: "Deepen Random Component References Module"
type: plan
status: draft
date: 2026-06-24
related:
  - docs/plans/random-component-parity-hardening-spec.md
  - docs/reference/workflow/PARITY_RANDOM_COMPONENT_SUITE.md
  - docs/adr/0010-random-component-parity-suite.md
---

# Deepen "Random Component References" — Architecture Deepening Plan

**Goal (per improve-codebase-architecture skill):** Turn the current shallow reference computation into a **deep module** with high **leverage** (callers get rich refs from tiny interface) and high **locality** (bugs, changes, and verification concentrate in one place).

Uses architecture terminology (Module, Interface, Seam, Depth, Leverage, Locality, Deletion test, Adapter) from the improve-codebase-architecture skill's `LANGUAGE.md` (`.agents/skills/improve-codebase-architecture/LANGUAGE.md`).

## Current State (Shallow)

The reference computation lives primarily inside `tests/support/random_component_parity.py` (still ~866 lines after Phase 1-3 extraction).

- `materialize_corpus`, `python_reference`, `_python_case_reference`, `_energy_samples`
- Low-level helpers (`_compare_values`, `_as_list`, etc.) are also here and pulled into `gate.py` / `diagnostics.py` via lazy imports.
- Circular dependency smell: package modules reach back into the big file.
- Deletion test: removing the file would force duplication of reference logic across the package, tests, and any future exact-proof code.

**Interface today** (what a caller must know):
- How to write TIFFs
- Exact shape of the manifest
- When to pass `include_hessian`
- Details of `_fourier_transform_input`, matching kernel loading, etc.
- MATLAB driver invocation for the "matlab" side

The interface is nearly as complex as the implementation → **shallow module**.

## Target: Deep "RandomComponentReferences" Module

**Proposed module** (new or extracted): `tests/support/random_component/references.py` (or a subpackage).

**Narrow Interface** (what callers actually need):

```python
def compute_python_references(
    manifest_path: Path, *, include_hessian: bool = False
) -> PythonRefs:  # structured dataclass or TypedDict with linspace + cases + energy

def compute_matlab_references(
    manifest_path: Path, matlab_mat_path: Path
) -> MatlabRefs: ...
```

Or even higher-leverage:

```python
def build_differential_inputs(
    corpus_manifest: CorpusManifest, mode: Literal["structural", "diagnostics"]
) -> DifferentialInputs:  # contains both sides + any needed metadata
```

**Seams** to introduce:
- `ReferenceProvider` seam (Python implementation vs. recorded fixture vs. future in-memory).
- `PaddedFFTProvider` internal seam (for testing the FFT-dependent parts without full pipeline).

**What moves behind the interface (implementation, not leaked):**
- TIFF materialization
- `_fourier_transform_input` + shape derivation
- Matching kernel loading + bessel/jv details
- Derivative kernel application + IFFT + principal energy sampling
- The exact 4 sample points chosen for Hessian
- All the "include_hessian" branching logic

## Benefits

**Leverage (for callers):**
- `gate.py` and `diagnostics.py` become dramatically simpler — they receive ready-to-compare structured refs.
- Future work (more sample points, additional energy kernels, recorded reference mode) only touches the References module.
- Tests for linspace/interp3/padded/energy can target the module directly without spinning up CLI or MATLAB driver.

**Locality (for maintainers):**
- All historical sources of ULP/scale surprises (IFFT floor, bessel drift, linspace 1-based, padded grid slicing) concentrate here.
- Deletion test now passes: removing References would re-create complexity in gate, diagnostics, tests, and any new proof surfaces.
- The main `random_component_parity.py` can stay a thin orchestrator + CLI.

## Migration Steps (Phased, low risk)

1. **Extract the module** (new file `tests/support/random_component/references.py`)
   - Move `materialize_corpus`, `python_reference`, `_python_case_reference`, `_energy_samples`, and supporting pure helpers.
   - Keep public functions with the same signatures initially for minimal disruption.

2. **Introduce typed shapes** (extend `models.py`)
   - `PythonRefs`, `MatlabRefs`, or a unified `DifferentialInputs`.
   - This gives the Interface something callers and tests can depend on.

3. **Update consumers**
   - `gate.py` and `diagnostics.py` import from the new module instead of reaching into `..._parity.py`.
   - Remove the lazy imports and circularity.

4. **Thin the orchestrator**
   - `run_differential` and `main` in `random_component_parity.py` become very small: materialize manifest → ask References module → ask Gate/Diagnostics → ask Reports → write artifacts.
   - `compare_references` (the compat shim) can also delegate to the new module.

5. **Add seams for testability**
   - Optional `ReferenceProvider` protocol so unit tests can inject pure-Python refs without any TIFF or MATLAB work.
   - This makes the "interface is the test surface" principle real for this area.

6. **Update docs and plan**
   - Refresh `PARITY_RANDOM_COMPONENT_SUITE.md` "How to hack" section.
   - Mark the original hardening spec's future extraction items as addressed by this deepening.

7. **Verify**
   - All 20+ unit tests still pass.
   - Full structural + diagnostics run against the Phase 0 baseline produces identical gate results.
   - Main file line count drops further.
   - Deletion test for the new References module passes.

## Risks & Mitigations

- **Temporary duplication during extraction**: Keep the old functions in `parity.py` as thin forwards for one commit if needed, then delete.
- **Signature changes**: Do them behind the new module first; the public `compare_references` / CLI entry points can stay stable.
- **Performance**: The heavy work (FFT + IFFTs) only happens when `include_hessian=True` — same as today. No regression.

## Relationship to Existing Work

This is the natural deepening of the Phase 1-3 hardening we just completed. The current split (gate / diagnostics / reports) is good, but the reference computation was left behind as the remaining source of complexity and circularity. Deepening it completes the vision of the original `random-component-parity-hardening-spec.md`.

It directly addresses the "shallow module" and "circular dependency" friction identified in the architecture review (see the generated HTML report).

## Recommendation

**Tackle this first.** Highest leverage for the random component area, directly reduces main file size, eliminates the cross-package import smell, and sets up any future work on the broader Parity Harness (see candidate 2 in the report) on a much cleaner foundation.

After this module exists, the next natural deepening is likely "ParityDifferentialHarness" (candidate 2) so that `analytics/parity/` code can depend on a narrow run interface instead of knowing the details of how references + gate + reports are wired.

---

**Status**: Ready to implement. Which part would you like to explore first (the References interface design, the seam for providers, or the impact on the broader harness)?