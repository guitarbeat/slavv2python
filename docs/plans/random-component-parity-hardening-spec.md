---
title: "Random Component Parity Suite — Implementation Hardening and Refactoring"
type: spec
status: draft
date: 2026-06-24
topic: random-component-parity-maintainability
related:
  - docs/reference/workflow/PARITY_RANDOM_COMPONENT_SUITE.md
  - docs/adr/0010-random-component-parity-suite.md
---

# Random Component Parity Suite — Implementation Hardening Spec

**Authoritative plan** for refactoring the implementation of the fast seeded MATLAB R2019a / Python random-component differential suite. Tasks live in this spec during active work; status and live runs live in [EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md) under the random component section. The user-facing workflow stays in [PARITY_RANDOM_COMPONENT_SUITE.md](../reference/workflow/PARITY_RANDOM_COMPONENT_SUITE.md).

---

## Summary

Refactor the Python implementation behind the Random Component Parity Suite so that it becomes dramatically simpler, smaller per file, more direct, and easier to extend — **without changing the observable behavior or exit code of the structural gate**.

This is a pure maintainability and architectural hardening effort. The suite remains a fast developer/CI diagnostic (ADR 0010) and is **not** part of crop or canonical certification claims.

---

## Motivation & Current Problems

The suite (described in PARITY_RANDOM_COMPONENT_SUITE.md) currently lives primarily in one file:

- `tests/support/random_component_parity.py` (~946 lines, 32 top-level functions after recent enhancements).
- It mixes corpus materialization, Python reference computation (using private energy kernels), MATLAB driver invocation, multi-phase comparison logic, ad-hoc dict-shaped reporting, summary formatting, and CLI.
- Recent useful additions ( `--mode structural|diagnostics`, explicit `structural_gate`, `query_kind` labels, `.txt` summary, CI unit precheck) were added by extending the same large module.

Key issues (observed during code review of the mode + reporting changes):

1. **File size & cohesion pressure**: The file is already near the repository 1000-line soft limit for Python modules. Every new diagnostic or guard pushes it closer.
2. **Ad-hoc data shapes**: The central report is a large nested dict (`passed`, `schema_version`, `structural_gate`, `linspace`, `hessian_diagnostics`, `cases[]`, `differences[]`, `first_difference`, etc.). Shape is only documented by code + hand-written test fixtures.
3. **Entangled structural vs advisory paths**:
   - `_hessian_diagnostics_for_case` is called unconditionally inside `compare_references`.
   - `run_differential` does post-construction mutation: `report["mode"] = ...; report["hessian_diagnostics"]["collected"] = ...`
   - A `collected` sentinel exists to let formatters explain "not present in this mode".
4. **White-box test coupling**: `tests/unit/parity/test_random_component_parity.py` imports a large number of `_`-prefixed symbols and reconstructs the exact internal report dict shape for every test.
5. **Control flow complexity**: Mix of early returns on first structural mismatch, size guards, and full traversal for diagnostics makes the "what actually gates" path harder to read than necessary.
6. **Future extension cost**: Adding another strictly-compared component or more sample points would require edits across many functions in the same file.

These are classic symptoms that a "code judo" restructuring (better models + separation of concerns) will pay off.

---

## Goals

- Make the **structural gate** logic small, direct, and obviously correct in isolation.
- Separate concerns cleanly:
  - Corpus & materialization
  - Reference computation (Python + loading MATLAB)
  - Structural comparison (the gate)
  - Advisory Hessian collection (only when requested)
  - Report assembly and formatting
  - Orchestration / CLI / MATLAB driver
- Introduce small, explicit data models (frozen dataclasses or well-documented TypedDicts + builders) so the report contract is readable and versionable.
- Keep every individual file well under the 1000-line guideline (target: most modules < 350–400 lines).
- Preserve **exact** structural gate semantics:
  - Same 128 linspace contexts (values + metadata)
  - Same 16 `interp3` queries per case (integer + half-integer + boundary/OOB)
  - Same `padded_shape_yxz`, sample `coordinate_yxz`, `valid` flags
  - Identical `passed` / `difference_count` / `first_difference` for the structural portions
  - Identical CLI exit code (0 on structural pass, 1 on fail)
- Make future additions (new fields, more samples, additional math kernels) touch only the relevant narrow modules.
- Improve the unit test surface so the majority of tests do not need private implementation details.

---

## Non-Goals

- Changing any numerical results or the set of values that participate in the strict structural compare.
- Turning the suite into a certification claim (it remains fast smoke + advisory ULP telemetry).
- Removing the need for MATLAB R2019a + `external/Vectorization-Public` for full differential runs.
- Major user-facing CLI or artifact format breakage (additive changes and a documented schema bump are acceptable if the structural gate behavior is identical).
- Moving the harness out of `tests/support` unless a clearly superior location is identified during refactoring.

---

## Requirements

**R1. Behavioral invariance (highest priority)**
- For `--mode structural`, the values of `report["passed"]`, `report["difference_count"]`, structural sections of `first_difference`, `linspace.passed`, and `structural_gate` must match a baseline run performed before the refactor.
- CLI exit status must be identical.

**R2. Report compatibility**
- Existing top-level keys consumed by the workflow, CI logs, `GITHUB_STEP_SUMMARY`, and `format_*_summary` helpers must continue to exist with the same meanings (or a clear migration path + `schema_version` bump).
- Per-case reports under `reports/*.json` must remain usable.

**R3. Testability**
- All 17+ existing unit tests (and new ones) must pass with `pytest -m "unit and parity"` without requiring MATLAB.
- The structural gate must be exercisable from tests using only public or narrowly scoped test helpers.

**R4. Size & structure**
- No single Python file in the implementation may grow past ~700 lines as a result of this work (strong preference for smaller focused modules).
- Clear module boundaries so a reader can answer "where is the structural gate logic?" in one hop.

**R5. Reproducibility**
- Full end-to-end runs (materialize + MATLAB driver + compare) on the same manifest must produce equivalent structural results before and after.

**R6. Documentation**
- The plan itself + updates to PARITY_RANDOM_COMPONENT_SUITE.md must describe the new internal structure at a level useful for future maintainers.

---

## Proposed Target Architecture

### 1. Module layout (suggested)

Keep a thin public entry point for backward compatibility with existing invocations and docs.

Preferred split (under `tests/support/`):

- `random_component_parity.py` — thin CLI wrapper + `run_differential` orchestration (small).
- `random_component/` (package) or sibling files:
  - `corpus.py` — `CorpusCase`, `load_manifest`, `materialize_corpus`, query generation, linspace context generation, manifest resolution.
  - `references.py` — `python_reference`, `_energy_samples` (or public equivalent), matching kernel loading, linspace evaluation using the promoted shims.
  - `compare.py` — low-level `_compare_values`, `_compare_sequence`, `_query_kind`, `compare_structural(...) -> StructuralResult`, `collect_hessian_diagnostics(...)`.
  - `report_models.py` — frozen dataclasses:
    - `StructuralGate`
    - `HessianSampleDiagnostics`
    - `CaseStructuralReport`
    - `RandomComponentReport` (top-level aggregate)
  - `matlab_driver.py` — `verify_matlab_prerequisites`, `run_matlab_driver`, `load_matlab_reference`.
  - `formatters.py` — `format_structural_summary`, `format_hessian_advisory_summary`, `write_case_reports`, `print_hessian...`.

Alternative (simpler if package overhead is unwanted): several focused files `random_component_*.py` imported by the main parity module. Either is acceptable as long as files stay small and responsibilities are clear.

### 2. Data models (key)

```python
@dataclass(frozen=True)
class StructuralGate:
    passed: bool
    linspace_context_count: int
    case_count: int
    query_count_per_case: int
    difference_count: int
    first_difference: Optional[dict] = None
    ...

@dataclass(frozen=True)
class HessianDiagnostics:
    collected: bool
    ...
```

The final JSON report can still be produced by a `to_dict()` or builder for compatibility.

### 3. Control flow (ideal)

```python
manifest = materialize_corpus(...)
py = python_reference(manifest, include_hessian=(mode=="diagnostics"))
mat = load_matlab_reference(...)

gate = compare_structural(py, mat, manifest=manifest)   # pure, no hessian work
hess = collect_hessian_diagnostics(py, mat) if mode == "diagnostics" else empty

report = assemble_report(gate, hess, mode=mode)
```

`compare_references` can remain (or delegate) during a transition period to avoid breaking callers.

### 4. Public vs internal surface

- Document a small stable public API for the harness (`run_differential`, the load_* helpers that tests legitimately need, the two formatters).
- Most comparison helpers become module-private inside the new structure.
- Unit tests primarily exercise the public surface + a few narrow "white box" helpers exposed explicitly for testing (e.g. `compare_structural`).

---

## Implementation Phases

### Phase 0 — Foundations (this spec + baseline)
- [x] Write and land this spec (commit `b9ddfe47`).
- [x] Captured known-good structural baseline (2026-06-24, on local MATLAB R2019a):
  - Command: `python -m tests.support.random_component_parity --output-dir workspace\scratch\random_component_baseline --matlab-exe "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" --mode structural`
  - Result: `passed: true`, `difference_count: 0`, `structural_gate.passed: true`.
  - All structural sections clean: 128 linspace contexts, 6 cases, 16 queries each (integer + half-integer + boundary/OOB).
  - Artifacts present:
    - `manifest.json`
    - `matlab_reference.mat`
    - `random_component_parity_report.json` + `.txt`
    - `reports/<case_id>.json` (6 files)
    - `inputs/*.tif` (6 materialized volumes)
  - This baseline will be used for equivalence verification after refactoring.
- [x] Added link from `PARITY_RANDOM_COMPONENT_SUITE.md` to this plan.
- [x] Ensured current unit tests are green: `pytest -m "unit and parity"` → 17 passed (see test run 2026-06-24).
- [x] (Optional) Reference added to EXACT_PROOF_FINDINGS.md.

### Phase 1 — Pure Structural Gate + Typed Models (strong version)
**Goal:** Make the structural gate a narrow, pure, first-class concept that has *zero knowledge* of Hessian/energy samples, mode flags, or advisory collection. Delete the entanglement instead of porting it.

- [x] Create `tests/support/random_component/models.py` (new focused file) + package with proper dataclasses:
  - `Mismatch`, `StructuralGateResult`
- [x] Implement `run_structural_gate(...) -> StructuralGateResult` in `tests/support/random_component/gate.py`.
  - Only structural checks. **Never** references energy/hessian.
- [x] Refactor `run_differential` to take the clean structural-only path for `--mode structural`.
  - No mutation, no "collected", no hessian call on structural path.
- [x] Structural path produces identical gate results vs Phase 0 baseline (verified via full MATLAB run).
- [x] Added focused test `test_run_structural_gate_produces_clean_result...` that asserts on `StructuralGateResult` and the clean report shape.
- [x] Full unit suite (now 18 tests) green.
- [x] Verified end-to-end with local MATLAB: structural gate matches baseline exactly (passed, diff_count=0, gate dicts, case results).
- [x] Refactored `compare_references` itself to delegate to clean gate + separate collector.
- [x] Removed dead `_hessian_diagnostics_for_case` (now unused after separation); main file back under 950 lines.
- [ ] (Future) Move low-level `_compare_*` helpers into the package; further shrink main file.

### Phase 2 — Hessian Collection & Advisory Path
- [x] Implemented `collect_hessian_diagnostics(...)` as completely separate module (`diagnostics.py`).
- [x] `run_differential` and `compare_references` now use gate + separate collector; hessian never called for structural.
- [x] Removed `collected` sentinel usage from structural path (structural reports set it false cleanly).
- [x] `format_hessian_advisory_summary` works with the collected data; structural path produces reports without hessian computation.
- [x] Even the legacy `compare_references` now uses clean separated components.

### Phase 3 — Reference Computation & Orchestration
- [x] Refactored `run_differential` and report assembly to always use clean gate + (optional) hessian, then package builders (`build_structural_report`, `build_diagnostics_report`).
- [x] `compare_references` (compat) also uses package builders for legacy shape.
- [x] Builders and orchestration live in / use the `random_component` package.
- [x] (Minor) `python_reference` / `_energy_samples` already conditional on include_hessian for heavy work; added comments for clarity. No unnecessary structures computed for structural path.
- [x] All reports remain byte-compatible with Phase 0 baseline for structural fields.
- [x] Main file shrunk to ~859 lines by moving builders to package.
- [x] Phase 3 complete per strong recommendations (orchestration uses models/builders, reference conditional clarified, main file leaner).

### Phase 4 — Test Surface & Compatibility
- [x] Added multiple tests exercising the new public API directly: run_structural_gate, collect_hessian_diagnostics, build_*_report (now 20 tests total).
- [x] compare_references remains as the compatibility shim (old signature, now internally delegates to clean gate+collector); no external scripts rely on it beyond the test file itself.
- [x] Ran full unit test suite (all green).
- [ ] (Partial) Private _ imports for specific low-level comparator unit tests remain (intentional white-box for _compare_values etc.); main test logic and new coverage now use public package surface heavily, reducing overall deep impl coupling.

### Phase 5 — End-to-End Verification with MATLAB
- [x] Fresh `--mode structural` run on same corpus (local MATLAB R2019a) → exit code 0, passed: true.
- [x] Key structural fields identical to Phase 0 baseline: passed, difference_count=0, structural_gate, linspace, first_difference, cases.
- [x] Optional `--mode diagnostics` run: hessian collected with mismatches (advisory), structural_gate still clean/passed.
- [x] CLI exit code: 0 for passed structural (1 would be for failure).
- [x] Artifacts confirmed usable: report.json, report.txt, manifest.json, reports/*.json (6), inputs/. All present and parseable.

### Phase 6 — Documentation, Polish, Landing
- [ ] Update `PARITY_RANDOM_COMPONENT_SUITE.md` with any new internal structure notes or "how to hack on the suite" guidance.
- [ ] Update docstrings, type hints, and the module docstring.
- [ ] Consider a small `schema_version` bump if the internal report assembly changed shape (keep old keys).
- [ ] Run ruff + mypy + full parity unit tests.
- [ ] Land the changes. Mark spec complete (or move residual items to TODO).

---

## Verification Strategy (Detailed)

1. **Baseline capture** (Phase 0): Successful `--mode structural` run + save of `random_component_parity_report.json` and the `.txt` summary + the resolved `manifest.json`.
2. **Structural equivalence** (Phase 5):
   - `report["passed"]` matches baseline.
   - `report["structural_gate"]["passed"]` matches.
   - `linspace` and case structural difference counts + first_difference (for structural components) are identical.
   - No new structural mismatches introduced.
3. **Advisory isolation**: In structural mode, `hessian_diagnostics.collected` is false (or the section is minimal) and no Hessian work influenced the gate decision.
4. **Unit tests**: All existing + new tests green without MATLAB.
5. **Full matrix**: At least one end-to-end structural run + one diagnostics run on the refactored code.

---

## Risks & Mitigations

- **MATLAB reproducibility**: Mitigated by using the exact same driver, manifest, and seeds. Always capture baseline on the machine that will also verify.
- **Report consumers outside this repo**: Document any schema change. Keep the structural sections stable.
- **Test breakage during refactor**: Keep a temporary delegating `compare_references` that builds the old dict shape from the new models until all call sites are updated.
- **Over-abstraction**: Prefer boring direct code. Models should be small and obvious. Do not introduce unnecessary layers just to have layers.
- **Time cost of full runs**: The corpus is tiny (6 × 16×32×32 volumes). Full structural runs are fast once MATLAB is warm.

---

## Success Criteria

- A maintainer can answer "what exactly causes a structural failure?" by reading < 100 lines of focused code.
- Adding a new strictly compared scalar (e.g. an additional energy field) requires changes in only the corpus case model + one comparison function + one place in the gate result.
- No Python source file in the suite implementation exceeds ~700 lines after the work (strong preference for much smaller).
- The refactored code passes a strict code-quality review (no ad-hoc dict growth, clear boundaries, logic lives in the right layer).
- Full verification run (local MATLAB) demonstrates identical structural gate outcome vs baseline.
- The suite is now pleasant to maintain and ready for the next useful diagnostic addition.

---

## Related Documents & Code

- [PARITY_RANDOM_COMPONENT_SUITE.md](../reference/workflow/PARITY_RANDOM_COMPONENT_SUITE.md) — operator workflow (update after refactor)
- [ADR 0010](../../adr/0010-random-component-parity-suite.md)
- Current implementation entry point: `tests/support/random_component_parity.py`
- MATLAB driver: `tests/support/matlab/random_component_reference.m`
- Export helpers: `export_random_linspace_overrides.py`, `export_random_matching_reference.py`
- Tests: `tests/unit/parity/test_random_component_parity.py`
- CI: `.github/workflows/matlab-random-component-parity.yml`

---

**This spec is the single source of truth for the hardening effort while it is active.**

Once complete, any future enhancements to the suite should reference the resulting structure and keep files small and focused.