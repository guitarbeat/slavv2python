# Requirements Document

## Introduction

This specification defines a behavior-preserving refactor plan for the largest
Python modules under `source/slavv/`.

The goal is not to redesign SLAVV's algorithms or public surfaces. The goal is
to make the codebase easier to reason about, safer to change, and easier to
test by splitting oversized modules into smaller ownership-aligned units.

Repository scan grounding for this spec identified these `source/slavv/` files
above 600 lines:

- `source/slavv/apps/web_app.py` - 2293 lines
- `source/slavv/core/edge_candidates.py` - 2184 lines
- `source/slavv/parity/comparison.py` - 1982 lines
- `source/slavv/visualization/network_plots.py` - 1794 lines
- `source/slavv/parity/metrics.py` - 1529 lines
- `source/slavv/core/energy.py` - 1218 lines
- `source/slavv/analysis/ml_curator.py` - 1046 lines
- `source/slavv/runtime/run_state.py` - 894 lines
- `source/slavv/parity/run_layout.py` - 788 lines
- `source/slavv/core/vertices.py` - 697 lines
- `source/slavv/analysis/geometry.py` - 654 lines
- `source/slavv/core/edge_selection.py` - 649 lines
- `source/slavv/parity/reporting.py` - 618 lines

## Problem Statement

The current modules are functional, but several are large enough that common
maintenance work now carries avoidable risk:

- related responsibilities are mixed together inside single files
- regression localization takes longer because many behaviors share one module
- type-check and test failures are more expensive to inspect
- algorithmic parity work and UI work both compete with structural complexity
- small bug fixes often require editing large, high-churn files

The refactor plan should reduce those risks without changing observable
behavior.

## Goals

- Reduce maintenance risk in the largest `source/slavv/` modules.
- Split modules along existing responsibility boundaries instead of task names.
- Preserve current CLI, app, and public import behavior unless explicitly
  documented otherwise.
- Keep parity-sensitive behavior stable and backed by focused regression tests.
- Sequence the work into small, reversible slices.

## Non-Goals

- Rewriting core vessel extraction algorithms.
- Changing the canonical staged comparison layout.
- Introducing a new CLI or app framework.
- Performing broad style cleanup unrelated to module extraction.
- Forcing every module under an arbitrary line limit in one pass.

## Refactor Principles

- Behavior first: structural edits should preserve outputs unless a bug fix is
  separately specified and tested.
- Small slices: each extraction should be independently testable and easy to
  revert.
- Stable surfaces: external callers should keep working through compatibility
  imports or wrapper functions during migration.
- Ownership alignment: new modules should reflect domain surfaces already
  present in `source/slavv/` and `dev/tests/`.
- Parity caution: changes in `source/slavv/core/` and `source/slavv/parity/`
  require stronger regression proof than pure UI or reporting moves.

## Requirements

### Requirement 1: Preserve Existing Behavior

**User Story:** As a maintainer, I want the refactor to preserve current
behavior, so that structural cleanup does not create algorithmic regressions.

#### Acceptance Criteria

1. Each refactor slice SHALL preserve existing public behavior unless a
   behavior change is explicitly called out in the slice plan.
2. Each extracted module SHALL be covered by targeted tests or existing
   regression coverage that proves unchanged behavior.
3. High-risk areas (`core`, `parity`, `runtime`) SHALL run focused regression
   coverage after each slice.

### Requirement 2: Preserve Public Entry Points

**User Story:** As a user of the package, I want existing CLI and import
surfaces to keep working, so that refactoring does not break callers.

#### Acceptance Criteria

1. Console entrypoints declared in `pyproject.toml` SHALL remain unchanged.
2. Existing import paths used by tests and documented workflows SHALL continue
   to resolve.
3. When code is moved, compatibility re-exports or thin wrapper functions SHALL
   be used until callers are updated intentionally.

### Requirement 3: Extract By Responsibility

**User Story:** As a contributor, I want oversized files split by responsibility,
so that I can find and edit the right logic without scanning a monolith.

#### Acceptance Criteria

1. New module boundaries SHALL follow real responsibility seams such as:
   - orchestration vs helpers
   - diagnostics vs mutation logic
   - plotting composition vs styling helpers
   - persistence/path handling vs status rendering
2. Extraction plans SHALL not create vague utility modules such as
   `misc_helpers.py` or `common.py` without a narrow documented purpose.
3. Test placement SHALL continue to mirror the owning package surface per
   `dev/tests/README.md`.

### Requirement 4: Prioritize Highest-Risk Files First

**User Story:** As a maintainer, I want refactor effort focused on the biggest
and riskiest modules first, so that the work reduces real maintenance pain.

#### Acceptance Criteria

1. The first implementation wave SHALL target the largest and highest-churn
   modules:
   - `source/slavv/core/edge_candidates.py`
   - `source/slavv/parity/comparison.py`
   - `source/slavv/apps/web_app.py`
   - `source/slavv/visualization/network_plots.py`
2. A later wave MAY cover smaller modules once the high-risk extractions prove
   the workflow.
3. Priority SHALL consider algorithmic criticality and change frequency, not
   only line count.

### Requirement 5: Keep Refactor Slices Reversible

**User Story:** As a maintainer, I want each extraction to be easy to validate
and revert, so that a bad slice does not stall unrelated work.

#### Acceptance Criteria

1. Each task slice SHALL define a narrow before/after scope.
2. Each slice SHALL avoid combining structural extraction with opportunistic
   feature work.
3. Each slice SHALL finish with a small verification command set.

### Requirement 6: Improve Internal Architecture Documentation

**User Story:** As a contributor, I want the post-refactor structure documented,
so that future work uses the new module boundaries consistently.

#### Acceptance Criteria

1. The design SHALL define the intended target module map for each oversized
   file.
2. Tasks SHALL include doc updates where a moved module changes contributor
   expectations.
3. Any compatibility shims SHALL be identified as temporary and tracked for
   later cleanup.

### Requirement 7: Use Canonical Repo Checks

**User Story:** As a maintainer, I want refactor validation to use the repo's
existing commands, so that proof is consistent with normal development.

#### Acceptance Criteria

1. Validation SHALL prefer the canonical commands in `AGENTS.md`.
2. Python refactor slices SHALL run `ruff`, targeted `pytest`, and `mypy` when
   the touched surfaces are in mypy-covered modules.
3. Full regression runs SHALL be scheduled at defined milestones rather than
   after every tiny extraction.
