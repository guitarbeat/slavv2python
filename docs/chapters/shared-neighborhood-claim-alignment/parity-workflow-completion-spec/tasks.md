# Implementation Plan: Parity Workflow Completion

## Overview

This spec now focuses on the last two unfinished backlog items from
`BOTTLENECK_TODO.md`:

1. Continue using shared-neighborhood diagnostics to drive the next
  `edge_candidates.py` or `edge_selection.py` iteration.
2. Promote a maintained proof artifact or report path for the stage-isolated
   `network` gate.

CLI reuse summaries and the stage-isolated `network` gate already exist in the
repo and are treated here as baseline dependencies. The remaining work is to
wire evidence-producing diagnostics into the current comparison flow and to
persist a durable proof surface for successful `network`-gate runs.

## Baseline Already In Place

- [x] Reuse-eligibility summaries are implemented in
      `source/slavv/parity/cli_summaries.py` and surfaced from the parity CLI.
- [x] Stage-isolated `network` gate validation and execution are implemented in
      `source/slavv/parity/network_gate.py`.
- [x] Comparison-mode reruns already force `comparison_exact_network=True` and
      persist network-gate execution metadata under `99_Metadata/`.
- [x] The staged run-root layout and replay-safe normalized params are already
      persisted and should remain the source of truth for new artifacts.
- [x] `dev/reports/tooling/README.md` classifies the March 23, 2026
      Ruff, mypy, and pytest outputs as archival snapshots rather than live
      repo status.

## Tasks

### Phase 1: Shared-Neighborhood Diagnostic Integration

- [x] 1. Define diagnostic report models and persistence contract
  - [x] 1.1 Create `source/slavv/parity/diagnostics.py`
    - Implement `NeighborhoodDivergence` dataclass for a single divergent
      neighborhood.
    - Implement `SharedNeighborhoodDiagnosticReport` dataclass for the complete
      report.
    - Include divergence categories aligned with the active backlog diagnosis:
      `claim_ordering`, `branch_invalidation`, and `partner_choice`.
    - Include severity classification, candidate-coverage deltas, and
      actionable recommendation fields.
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.9_

  - [x]* 1.2 Add unit tests for diagnostic models
    - Verify dataclass defaults and serialization-friendly structure.
    - Verify divergence-category and severity handling.
    - _Requirements: 1.2, 1.3, 1.4, 1.5_

- [x] 2. Implement shared-neighborhood report generation
  - [x] 2.1 Add `generate_shared_neighborhood_diagnostics()`
    - Load the existing MATLAB and Python edge/candidate artifacts from the
      staged run root.
    - Compare neighborhood-level claim ordering, branch invalidation behavior,
      and local partner choice.
    - Quantify candidate-coverage deltas so counts remain a first triage
      signal without becoming the only signal.
    - Produce recommendation text that points investigators at likely
      `source/slavv/core/edge_candidates.py` and
      `source/slavv/core/edge_selection.py` surfaces.
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.9_

  - [x]* 2.2 Add unit tests for diagnostic generation
    - Cover claim-ordering, branch-invalidation, and partner-choice detection.
    - Cover candidate-coverage delta calculation.
    - Cover recommendation generation from representative divergence patterns.
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.9_

- [x] 3. Persist diagnostics and surface them in the current workflow
  - [x] 3.1 Persist canonical report artifacts
    - Write JSON to
      `03_Analysis/shared_neighborhood_diagnostics.json`.
    - Write a human-readable markdown companion to
      `03_Analysis/shared_neighborhood_diagnostics.md`.
    - Keep filenames stable so downstream docs and investigators can reference
      them consistently.
    - _Requirements: 1.7, 1.8_

  - [x] 3.2 Integrate diagnostic recommendation and summary into
          `source/slavv/parity/comparison.py`
    - Recommend diagnostics when edge parity fails and the current workflow
      state makes the shared-neighborhood report actionable.
    - Prefer the recommendation when the stage-isolated `network` gate
      succeeds, because that isolates the remaining gap to `edges`.
    - Print a compact CLI summary of the top divergence patterns when a report
      already exists for the run root.
    - _Requirements: 1.1, 1.7, 1.8, 1.9_

  - [x]* 3.3 Add integration tests for diagnostic workflow behavior
    - Verify recommendation text after an edge parity gap.
    - Verify report persistence into `03_Analysis/`.
    - Verify CLI summary rendering from an existing report.
    - _Requirements: 1.1, 1.7, 1.8_

- [x] 4. Phase 1 checkpoint
  - Confirm diagnostics produce actionable, neighborhood-level evidence rather
    than only downstream noise.
  - Confirm the report points investigators at concrete next code surfaces.
  - Ask the user if questions arise.

### Phase 2: Maintained Network-Gate Proof Artifacts

- [x] 5. Define proof artifact models around existing network-gate metadata
  - [x] 5.1 Create `source/slavv/parity/proof_artifacts.py`
    - Implement `StageIsolatedNetworkProof` dataclass.
    - Implement `ProofArtifactIndex` dataclass.
    - Model provenance from the staged MATLAB batch plus fingerprints from the
      existing network-gate execution flow.
    - Record exact parity status for vertices, edges, and strands.
    - Keep resource usage optional so the proof format is portable on Windows.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 2.9_

  - [x]* 5.2 Add unit tests for proof models
    - Verify serialization-friendly structure and optional resource fields.
    - Verify parity-status and provenance fields.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 6. Generate proof artifacts from successful `network`-gate runs
  - [x] 6.1 Implement `generate_proof_artifact()`
    - Consume the persisted `network_gate_execution.json` data and the current
      run layout instead of duplicating gate logic.
    - Persist a canonical JSON proof artifact under
      `03_Analysis/proof_artifacts/network_gate_proof_{timestamp}.json`.
    - Persist a matching markdown proof artifact under
      `03_Analysis/proof_artifacts/network_gate_proof_{timestamp}.md`.
    - Include provenance, input/output fingerprints, exact parity status, and
      elapsed timing.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [x] 6.2 Implement `maintain_proof_artifact_index()`
    - Persist `03_Analysis/proof_artifact_index.json`.
    - Track all successful proofs chronologically plus the latest proof
      reference.
    - Rebuild the index from existing proof artifacts if the index is missing
      or corrupt.
    - _Requirements: 2.8, 2.9_

  - [x]* 6.3 Add unit tests for proof generation and indexing
    - Verify proof generation from network-gate execution metadata.
    - Verify index updates and rebuild behavior.
    - _Requirements: 2.1, 2.7, 2.8, 2.9_

- [x] 7. Integrate proof generation into the existing `network`-gate workflow
  - [x] 7.1 Update `source/slavv/parity/network_gate.py`
    - Generate proof artifacts only after successful stage-isolated
      `network`-gate execution.
    - Skip proof generation when the gate fails or does not achieve the proof
      preconditions.
    - Surface a compact proof summary back to the CLI output.
    - _Requirements: 2.1, 2.7, 2.8_

  - [x] 7.2 Add CLI proof display support
    - Implement `display_latest_proof_summary()` in
      `source/slavv/parity/proof_artifacts.py`.
    - Add `slavv parity-proof --run-dir <path>` to
      `source/slavv/apps/parity_cli.py`.
    - Display the latest proof summary and list known proof artifacts with
      timestamps.
    - _Requirements: 2.10_

  - [x]* 7.3 Add integration tests for proof workflow behavior
    - Verify proof generation after successful `network`-gate runs.
    - Verify proof suppression on failed gate runs.
    - Verify CLI proof display against persisted artifacts.
    - _Requirements: 2.1, 2.7, 2.8, 2.10_

- [x] 8. Phase 2 checkpoint
  - Confirm a maintained proof path exists under `03_Analysis/`.
  - Confirm repeated successful gate runs update the proof index cleanly.
  - Ask the user if questions arise.

### End-to-End Validation

- [x] 9. Validate the remaining parity workflow end to end
  - [x]* 9.1 Add a diagnostic workflow test
    - Start from a run root that exhibits an `edges` parity gap.
    - Verify diagnostic recommendation, report generation, and report
      persistence.
    - Verify the stage-isolated `network` gate can still produce a proof
      artifact when exact MATLAB edges are imported.
    - _Requirements: 1.1, 2.1_

  - [x] 9.2 Run final verification for the narrowed scope
    - Run the relevant unit/integration tests for diagnostics and proof
      artifacts.
    - Run `python -m pytest -m "unit or integration"` if the change crosses
      module boundaries.
    - Run `python -m ruff format source tests`, `python -m ruff check source tests --fix`,
      and `python -m mypy`.
    - Verify staged artifact persistence stays within `03_Analysis/` and
      `99_Metadata/` conventions.
    - _Requirements: 1.1, 2.1_

- [x] 10. Reconcile archived tooling context for touched scope
  - [x] 10.1 Review `dev/reports/tooling/`
    - Treat the archived Ruff, mypy, and pytest outputs as historical context,
      not as current pass/fail gates.
    - Note any overlap between touched files and archived findings before
      implementation is treated as complete.
    - Current workspace note: `dev/reports/tooling/` is not present in this
      checkout; historical-tooling context was treated as archival requirement,
      while verification used current canonical gates.
    - _Requirements: 3.1, 3.3_

  - [x] 10.2 Verify touched modules do not widen historical tooling debt
    - Confirm new diagnostics/proof files are Ruff- and mypy-clean.
    - Confirm touched integration points do not introduce new pytest collection
      issues.
    - Narrow historical issues opportunistically only where the implementation
      already touches the file.
    - _Requirements: 3.2, 3.3, 3.4_

- [x] 11. Final checkpoint
  - Ensure the last two backlog items from `BOTTLENECK_TODO.md` are covered by
    implemented code, persisted artifacts, and regression tests.
  - Verify the spec no longer treats already-landed CLI summary and
    `network`-gate foundations as unfinished primary work.
  - Verify the archived tooling snapshots were accounted for without expanding
    the scope into a repo-wide cleanup project.
  - Ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for a
  narrower MVP, but the comparison and proof integrations should still receive
  at least one integration-level verification pass before the work is treated
  as complete.
- The active parity diagnosis still lives at the `edges` stage. These tasks are
  intended to improve evidence and iteration speed, not to replace the actual
  `edge_candidates.py` / `edge_selection.py` convergence work.
- All new artifacts must preserve the canonical staged layout:
  `01_Input/`, `02_Output/`, `03_Analysis/`, `99_Metadata/`.
- New modules should follow existing repository conventions:
  `from __future__ import annotations`, `pathlib.Path`, explicit encodings, and
  `logging` instead of `print()` in library code.

