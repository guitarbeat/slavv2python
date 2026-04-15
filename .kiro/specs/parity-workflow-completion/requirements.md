# Requirements Document

## Introduction

This specification now covers the remaining two unfinished workflow
improvements from `BOTTLENECK_TODO.md`:

1. Continued use of shared-neighborhood diagnostics to drive the next
   `edge_candidates.py` or `tracing.py` iteration.
2. Promotion of a maintained proof artifact or report path for the
   stage-isolated `network` gate.

CLI reuse summaries and the stage-isolated `network` gate are already present
in the codebase and are treated here as required baseline dependencies rather
than primary unfinished scope.

## Current Baseline

- Reuse-eligibility summaries already exist and should remain the current
  source of workflow guidance.
- Stage-isolated `network`-gate validation and execution already exist and
  should remain the current source of downstream isolation evidence.
- Staged-layout persistence under `01_Input/`, `02_Output/`, `03_Analysis/`,
  and `99_Metadata/` remains the canonical artifact contract.
- `workspace/reports/tooling/` contains archived March 23, 2026 Ruff, mypy,
  and pytest snapshots. These are reference artifacts, not live status.

## Glossary

- **CLI**: Command-line interface for the parity comparison tool
- **Parity_Workflow**: The complete MATLAB vs Python comparison workflow system
- **Stage_Isolated_Network_Gate**: A workflow loop that imports exact MATLAB
  edges and reruns only Python network assembly to isolate parity issues
- **Shared_Neighborhood_Diagnostics**: Diagnostic tools that analyze
  neighborhood-level claim ordering and partner-choice differences between
  MATLAB and Python
- **Edge_Candidates**: Python module responsible for candidate-edge generation
  during tracing
- **Tracing**: Python module responsible for edge selection and tracing logic
- **Proof_Artifact**: A maintained report or evidence file demonstrating that
  the stage-isolated `network` gate produces exact parity
- **MATLAB_Batch**: A completed MATLAB run output that can be imported and
  reused by Python

## Requirements

### Requirement 1: Shared-Neighborhood Diagnostic Integration

**User Story:** As a developer, I want shared-neighborhood diagnostics to drive
the next iteration of `edge_candidates.py` or `tracing.py`, so that I can
systematically address parity gaps using evidence-based insights instead of
relying on downstream noise.

#### Acceptance Criteria

1. WHEN a parity gap is detected at the `edges` stage, THE Parity_Workflow
   SHALL recommend running shared-neighborhood diagnostics
2. THE Shared_Neighborhood_Diagnostics SHALL generate a report identifying
   neighborhood-level claim ordering differences
3. THE Shared_Neighborhood_Diagnostics SHALL generate a report identifying
   branch invalidation differences
4. THE Shared_Neighborhood_Diagnostics SHALL generate a report identifying
   local partner-choice differences
5. THE Diagnostic_Report SHALL include specific vertex IDs or neighborhood
   identifiers where MATLAB and Python diverge
6. THE Diagnostic_Report SHALL quantify candidate-coverage differences between
   MATLAB and Python
7. WHEN diagnostic reports are available, THE CLI SHALL display a compact
   summary of the top divergence patterns
8. THE Diagnostic_Report SHALL persist under `03_Analysis/` in the staged run
   layout as both machine-readable and human-readable artifacts
9. THE Diagnostic_Report SHALL include actionable recommendations for which
   `Edge_Candidates` or `Tracing` code surfaces to investigate next

### Requirement 2: Maintained Stage-Isolated Network Proof Artifacts

**User Story:** As a developer, I want a maintained proof artifact for the
stage-isolated `network` gate, so that I can demonstrate and track that
network assembly achieves exact parity when given exact MATLAB edges.

#### Acceptance Criteria

1. THE Parity_Workflow SHALL generate a
   `Stage_Isolated_Network_Proof_Report` after successful stage-isolated
   `network`-gate execution
2. THE Stage_Isolated_Network_Proof_Report SHALL document the input MATLAB
   batch provenance
3. THE Stage_Isolated_Network_Proof_Report SHALL document the exact parity
   status for vertices, edges, and strands
4. THE Stage_Isolated_Network_Proof_Report SHALL include execution timing and
   any available resource-usage metadata without requiring platform-specific
   metrics
5. THE Stage_Isolated_Network_Proof_Report SHALL include checksums or
   fingerprints of the imported MATLAB artifacts used by the gate
6. THE Stage_Isolated_Network_Proof_Report SHALL include checksums or
   fingerprints of the resulting Python outputs used to confirm parity
7. THE Stage_Isolated_Network_Proof_Report SHALL persist under `03_Analysis/`
   with canonical filenames in both JSON and human-readable Markdown
8. WHEN multiple successful stage-isolated `network`-gate runs exist, THE
   Parity_Workflow SHALL maintain a proof artifact index
9. THE Proof_Artifact_Index SHALL list successful proof artifacts with their
   timestamps, parity status, and latest-proof reference
10. THE CLI SHALL provide a command to display the latest proof artifact
    summary for a run root

### Requirement 3: Tooling Debt Handling For Touched Scope

**User Story:** As a developer, I want the remaining parity workflow work to
respect existing tooling debt without turning this spec into a repo-wide cleanup
project, so that diagnostics and proof artifacts can land cleanly.

#### Acceptance Criteria

1. THE Parity_Workflow completion work SHALL treat
   `workspace/reports/tooling/` as historical reference only, not as the
   canonical source of current repo health
2. THE implementation for diagnostics and proof artifacts SHALL avoid
   introducing new Ruff, mypy, or pytest issues in touched modules and tests
3. WHEN the implementation touches a file that appears in the archived tooling
   snapshots, THE change SHALL either preserve behavior without widening the
   recorded issue surface or intentionally narrow that issue surface
4. THE final verification pass for this spec SHALL run the repo's canonical
   commands from `AGENTS.md` for the touched scope, even if the archived
   snapshots contain broader historical failures
5. THIS spec SHALL NOT require fixing all archived repo-wide tooling failures
   outside the diagnostics/proof-artifact implementation path

## Out Of Primary Scope

- Re-implementing reuse-eligibility summaries that already exist in the repo
- Re-implementing the stage-isolated `network` gate itself unless small wiring
  changes are required to support diagnostics or proof generation
- Solving the underlying `edges` parity gap directly inside this spec; this
  spec improves iteration evidence and proof surfaces around that work
