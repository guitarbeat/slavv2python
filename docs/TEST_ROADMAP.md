# Test Roadmap

This roadmap prioritizes fewer, higher-signal tests over broad, repetitive coverage.

## Goals

- Reduce test maintenance cost.
- Improve failure signal quality.
- Keep confidence in critical algorithms and I/O paths.
- Stabilize CI behavior across Windows/Linux.

## Success Metrics

- Runtime: reduce default `pytest tests/` wall time by 30-40%.
- Reliability: eliminate non-deterministic temp-path and filesystem failures in CI.
- Signal: each failing test should map to one clear behavior regression.
- Coverage quality: track behavior coverage for core modules instead of raw line count.

## Phase 0 (Now): Baseline and Guardrails

- Record current baseline:
  - total tests
  - suite runtime
  - top 10 slowest tests (`pytest --durations=10`)
  - flaky/infra-failing tests by platform
- Define test categories in markers:
  - `unit`
  - `integration`
  - `ui`
  - `slow`
  - `regression`
- Add CI split:
  - required fast lane (`unit`, selected `integration`)
  - optional/nightly full lane (all tests)

## Phase 1: Remove Low-Value Redundancy

- Collapse repetitive tests into scenario-driven parameterized tests.
- Remove assertions that only restate implementation details without behavior value.
- Replace random-data tests with deterministic fixtures.
- Keep a single canonical test per behavior branch.

## Phase 2: Stabilize Environment-Sensitive Tests

- Standardize temporary directory handling for tests that write files.
- Isolate Windows-specific path/ACL-sensitive tests behind clear fixtures.
- Separate true product failures from environment setup failures in diagnostics.
- Add explicit skip/xfail only when platform limitations are documented.

## Phase 3: Strengthen High-Value Regression Coverage

- Focus regression suites on:
  - core tracing/energy outputs
  - network export/import round-trips (MAT/CASX/VMV/JSON/CSV)
  - MATLAB parity checks for critical metrics
- Prefer fixture-based synthetic datasets with known expected outputs.
- Add one end-to-end smoke test per major workflow (CLI, API, UI entry point).

## Phase 4: CI and Developer Experience

- Add a `quick` local command for pre-commit confidence (fast lane only).
- Publish failing-test triage guidelines:
  - behavior regression
  - test defect
  - infra/environment defect
- Report test trends in CI artifacts:
  - runtime trend
  - flaky test count
  - failure category breakdown

## Ownership and Cadence

- Review roadmap progress every 2 weeks.
- Each PR touching core algorithms should include:
  - at least one behavior-level test update
  - runtime impact note when tests are added

## Immediate Next Steps

1. Add pytest markers in `pyproject.toml`.
2. Create fast-lane CI job and nightly full-suite job.
3. Audit top 5 slowest and top 5 flakiest modules.
4. Continue refactoring large repetitive test modules to scenario-driven format.
