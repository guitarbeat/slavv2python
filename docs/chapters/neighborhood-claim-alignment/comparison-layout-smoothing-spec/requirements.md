# Requirements Document

## Introduction

This specification defines a smoother comparison-run layout for `slavv_comparisons/`
that improves discoverability, cleanup safety, and cross-run comparison speed
without breaking the existing staged run contract.

The baseline staged folders remain unchanged:

- `01_Input/`
- `02_Output/`
- `03_Analysis/`
- `99_Metadata/`

This spec adds an organization layer above run roots and lightweight status
metadata to support efficient references and consolidation workflows.

## Problem Statement

Current run folders are usable but harder to navigate at scale because:

- run roots are mostly flat and not grouped by experiment intent
- canonical references are implicit instead of explicit
- superseded or failed runs are not uniformly tagged for safe cleanup
- quick comparison across runs requires opening many files manually

Repository scan also shows mixed run-root topology that the plan must support:

- direct staged roots (for example `20260327_150656_clean_parity`)
- aggregate roots that contain `run_*` children instead of direct staged folders
   (for example `20260328_142659_python_consistency`)

## Goals

- Preserve current staged artifact layout compatibility.
- Add minimal structure for faster run discovery and references.
- Make cleanup decisions scriptable and low-risk.
- Keep one-off manual workflows possible.

## Non-Goals

- Rewriting parity algorithms or comparison metrics.
- Replacing existing staged layout semantics.
- Introducing mandatory external tracking services.

## Glossary

- **Run Root**: A single comparison execution directory containing staged
  folders (`01_Input/`, `02_Output/`, `03_Analysis/`, `99_Metadata/`).
- **Experiment Slug**: A stable grouping name for related runs.
- **Pointer File**: A small file that stores the path to a key run root
  (for example latest completed or canonical acceptance).
- **Run Status Metadata**: Machine-readable lifecycle/state fields for a run.

## Requirements

### Requirement 1: Experiment Grouping Layer

**User Story:** As a developer, I want run roots grouped by experiment intent,
so that I can quickly find comparable runs without scanning a large flat list.

#### Acceptance Criteria

1. The layout SHALL support this path pattern:
   `slavv_comparisons/experiments/<experiment_slug>/runs/<run_root_name>/`.
2. Existing staged run internals SHALL remain unchanged inside each run root.
3. The system SHALL allow legacy top-level run roots during migration.
4. The system SHALL provide a per-experiment index artifact for run summaries.

### Requirement 2: Canonical Pointer Files

**User Story:** As a developer, I want explicit pointer files for key run roots,
so that chapter docs and operators can jump to canonical outputs instantly.

#### Acceptance Criteria

1. The layout SHALL include `slavv_comparisons/pointers/`.
2. Pointer files SHALL include at least:
   - `latest_completed.txt`
   - `canonical_acceptance.txt`
   - `best_saved_batch.txt`
3. Each pointer file SHALL contain exactly one repo-relative run-root path.
4. Pointer updates SHALL be explicit and human-auditable.

### Requirement 3: Run Lifecycle Metadata

**User Story:** As a developer, I want lifecycle metadata in each run,
so that I can automate cleanup and supersession decisions safely.

#### Acceptance Criteria

1. Each managed run root SHALL include `99_Metadata/status.json`.
2. `status.json` SHALL include fields:
   - `state` (`completed`, `failed`, `incomplete`, `superseded`, `archived`)
   - `supersedes` (optional path)
   - `superseded_by` (optional path)
   - `retention` (`keep`, `eligible_for_cleanup`, `archive`)
   - `quality_gate` (`pass`, `fail`, `partial`, `unknown`)
3. Failed runs SHALL be eligible for cleanup only when a replacement exists
   or retention explicitly allows removal.
4. Canonical runs referenced by pointer files SHALL default to `retention=keep`.
5. For aggregate roots that contain `run_*` children, status metadata MAY be
   stored at the aggregate root or at each child run, but the chosen policy
   SHALL be consistent and documented.

### Requirement 4: Date-First Naming Convention

**User Story:** As a developer, I want consistent date-first names,
so that run roots sort naturally and are easier to scan.

#### Acceptance Criteria

1. New run roots SHALL use one of:
   - `YYYYMMDD_HHMMSS_<label>`
   - `YYYYMMDD_<label>`
2. New roots SHALL NOT use suffix-date names (for example `<label>_YYYYMMDD`).
3. Migration tooling SHALL support one-off renames from suffix-date form.

### Requirement 5: Backward-Compatible Consolidation

**User Story:** As a maintainer, I want consolidation to be low-risk,
so that current workflows and docs are not disrupted.

#### Acceptance Criteria

1. Consolidation tooling SHALL operate in a non-destructive mode first
   (detect/report before delete).
2. Link-bearing docs SHALL be updated when run-root names change.
3. Empty directories created by normalization moves SHALL be removed.
4. The staged run contract described in
   `docs/reference/COMPARISON_LAYOUT.md` SHALL remain authoritative.
5. Deletion of non-empty run roots SHALL require an explicit allow-list or
   equivalent positive confirmation input.

### Requirement 6: Reference Efficiency

**User Story:** As a chapter author, I want compact machine-readable indexes,
so that I can compare run outcomes without opening many files.

#### Acceptance Criteria

1. Each experiment index SHALL summarize at minimum:
   - run path
   - timestamp
   - state
   - key parity counts/status (if available)
2. Indexes SHALL be JSON for scriptability.
3. Index generation SHALL tolerate missing optional artifacts gracefully.

### Requirement 7: Execution Robustness And Weakness Mitigation

**User Story:** As a maintainer, I want the consolidation workflow to be robust
against known operational weaknesses, so that migration runs are repeatable and
auditable.

#### Acceptance Criteria

1. Consolidation operations SHALL be implemented as script files under
   `dev/scripts/maintenance/` instead of fragile long shell one-liners.
2. The workflow SHALL emit a machine-readable migration report describing:
   proposed moves, applied moves, skips, conflicts, and deletions.
3. After any rename or move, the workflow SHALL run a doc-reference verification
   step and report stale references.
4. Status inference SHALL use precedence rules that tolerate partial metadata
   (for example missing root `run_snapshot.json` for aggregate roots).
5. The workflow SHALL support idempotent re-run behavior: re-running after a
   partial migration SHALL not duplicate or corrupt layout state.
