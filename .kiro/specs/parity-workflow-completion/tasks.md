# Implementation Plan: Parity Workflow Completion

## Overview

This implementation follows a 4-phase migration path that builds on existing workflow assessment infrastructure to deliver CLI reuse eligibility summaries, stage-isolated network gate reliability, shared-neighborhood diagnostic integration, and proof artifact promotion. Each phase delivers incremental value while maintaining backward compatibility with existing parity workflow tooling.

## Tasks

### Phase 1: CLI Reuse Eligibility Summaries

- [ ] 1. Extend workflow assessment data model
  - [x] 1.1 Add new fields to LoopAssessmentReport dataclass
    - Add `reuse_commands`, `missing_artifacts`, `artifact_explanations`, and `next_action_commands` fields with default factories
    - Ensure backward compatibility with existing LoopAssessmentReport usage
    - _Requirements: 1.2, 1.3, 1.4, 1.6, 1.7_

  - [ ]* 1.2 Write unit tests for enhanced LoopAssessmentReport
    - Test dataclass instantiation with new fields
    - Test default factory behavior
    - Test serialization/deserialization if applicable
    - _Requirements: 1.2, 1.3, 1.4_

- [ ] 2. Implement CLI summary formatting module
  - [x] 2.1 Create source/slavv/parity/cli_summaries.py module
    - Implement `format_reuse_eligibility_summary()` function
    - Implement `generate_reuse_commands()` function
    - Implement missing artifact explanation formatting
    - Implement next action command generation
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

  - [ ]* 2.2 Write unit tests for CLI summary formatting
    - Test summary formatting with various loop assessment states
    - Test command generation for analysis-only comparison
    - Test command generation for imported-MATLAB edge rerun
    - Test command generation for stage-isolated network gate
    - Test missing artifact explanation formatting
    - Test next action recommendation generation
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

- [ ] 3. Integrate CLI summaries into parity CLI
  - [x] 3.1 Modify source/slavv/apps/parity_cli.py to display summaries
    - Add summary display after successful comparison run
    - Add summary display after --validate-only run
    - Add summary display after --resume-latest compatibility check
    - _Requirements: 1.1, 1.5, 1.7, 1.8_

  - [ ]* 3.2 Write integration tests for CLI summary display
    - Test complete flow from loop assessment to CLI summary display
    - Test reuse command generation with actual run root
    - Test missing artifact detection with partial run state
    - Test next action recommendation with various workflow states
    - _Requirements: 1.1, 1.5, 1.7, 1.8_

- [x] 4. Phase 1 Checkpoint
  - Ensure all Phase 1 tests pass
  - Verify CLI summaries display correctly for various workflow states
  - Ask the user if questions arise

### Phase 2: Stage-Isolated Network Gate Reliability

- [ ] 5. Implement network gate validation module
  - [x] 5.1 Create source/slavv/parity/network_gate.py module
    - Implement `NetworkGateValidation` dataclass
    - Implement `validate_network_gate_artifacts()` function
    - Implement artifact fingerprint generation using hashlib
    - Implement fast-fail validation with specific error messages
    - _Requirements: 2.2, 2.3, 2.8_

  - [ ]* 5.2 Write unit tests for network gate validation
    - Test validation with complete artifacts
    - Test validation failure with missing MATLAB edges
    - Test validation failure with missing MATLAB vertices
    - Test validation failure with missing MATLAB energy
    - Test fingerprint generation for artifacts
    - Test validation error message formatting
    - _Requirements: 2.2, 2.3, 2.8_

- [ ] 6. Implement network gate execution logic
  - [x] 6.1 Add network gate executor to source/slavv/parity/network_gate.py
    - Implement `NetworkGateExecution` dataclass
    - Implement `execute_stage_isolated_network_gate()` function
    - Add timing measurement for execution phases
    - Force comparison_exact_network=True mode
    - Import MATLAB edges/vertices/energy artifacts
    - Rerun Python from network stage
    - Compare results and record parity status
    - Persist execution metadata to 99_Metadata/network_gate_execution.json
    - _Requirements: 2.1, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [ ]* 6.2 Write unit tests for network gate execution components
    - Test NetworkGateExecution dataclass
    - Test timing measurement logic
    - Test parity status recording
    - Test execution metadata serialization
    - _Requirements: 2.1, 2.5, 2.6_

- [ ] 7. Integrate network gate into comparison workflow
  - [ ] 7.1 Modify source/slavv/parity/comparison.py for network gate integration
    - Add validation call before network gate execution
    - Add network gate execution for --python-parity-rerun-from network
    - Add parity status reporting to CLI output
    - _Requirements: 2.2, 2.3, 2.5, 2.7_

  - [ ]* 7.2 Write integration tests for network gate flow
    - Test complete network gate execution with imported MATLAB artifacts
    - Test fast-fail validation with missing artifacts
    - Test timing measurement and 30s performance target
    - Test parity status reporting
    - Test execution metadata persistence
    - _Requirements: 2.1, 2.2, 2.3, 2.5, 2.6_

- [ ] 8. Add network gate performance validation
  - [ ]* 8.1 Write diagnostic tests for network gate performance
    - Test network gate completes in <30s for standard test volume
    - Test validation overhead is <1s
    - Test execution remains isolated from edge candidate changes
    - _Requirements: 2.1, 2.7_

- [ ] 9. Phase 2 Checkpoint
  - Ensure all Phase 2 tests pass
  - Verify network gate executes reliably and meets performance targets
  - Ask the user if questions arise

### Phase 3: Shared-Neighborhood Diagnostic Integration

- [ ] 10. Implement diagnostic data models
  - [ ] 10.1 Create source/slavv/parity/diagnostics.py module
    - Implement `NeighborhoodDivergence` dataclass
    - Implement `SharedNeighborhoodDiagnosticReport` dataclass
    - Add divergence type classification (claim_ordering, branch_invalidation, partner_choice)
    - Add severity classification (high, medium, low)
    - _Requirements: 3.2, 3.3, 3.4, 3.5_

  - [ ]* 10.2 Write unit tests for diagnostic data models
    - Test NeighborhoodDivergence dataclass
    - Test SharedNeighborhoodDiagnosticReport dataclass
    - Test divergence type classification
    - Test severity classification
    - _Requirements: 3.2, 3.3, 3.4, 3.5_

- [ ] 11. Implement diagnostic report generation
  - [ ] 11.1 Add diagnostic generator to source/slavv/parity/diagnostics.py
    - Implement `generate_shared_neighborhood_diagnostics()` function
    - Implement neighborhood-level claim ordering comparison
    - Implement branch invalidation difference detection
    - Implement partner choice divergence detection
    - Implement candidate coverage delta calculation
    - Generate actionable recommendations from divergence patterns
    - Identify specific edge_candidates.py and tracing.py code to review
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 3.9_

  - [ ]* 11.2 Write unit tests for diagnostic generation
    - Test neighborhood divergence detection logic
    - Test claim ordering comparison
    - Test branch invalidation detection
    - Test partner choice divergence detection
    - Test candidate coverage delta calculation
    - Test recommendation generation from divergence patterns
    - Test diagnostic report serialization
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 3.9_

- [ ] 12. Implement diagnostic persistence and CLI integration
  - [ ] 12.1 Add diagnostic persistence to source/slavv/parity/diagnostics.py
    - Persist diagnostic reports to 03_Analysis/shared_neighborhood_diagnostics.json
    - Generate human-readable markdown report at 03_Analysis/shared_neighborhood_diagnostics.md
    - Implement `recommend_diagnostics_if_needed()` function
    - _Requirements: 3.1, 3.7, 3.8, 3.9_

  - [ ] 12.2 Integrate diagnostics into comparison workflow
    - Add diagnostic recommendation after edge parity gap detection
    - Display diagnostic command when edges parity fails but network parity succeeds
    - Display diagnostic summary when reports are available
    - _Requirements: 3.1, 3.7_

  - [ ]* 12.3 Write integration tests for diagnostic integration
    - Test diagnostic recommendation after parity gap detection
    - Test diagnostic report generation with real edge artifacts
    - Test diagnostic report persistence to 03_Analysis/
    - Test CLI diagnostic summary display
    - _Requirements: 3.1, 3.7, 3.8_

- [ ] 13. Phase 3 Checkpoint
  - Ensure all Phase 3 tests pass
  - Verify diagnostic reports generate actionable insights
  - Ask the user if questions arise

### Phase 4: Proof Artifact Promotion

- [ ] 14. Implement proof artifact data models
  - [ ] 14.1 Create source/slavv/parity/proof_artifacts.py module
    - Implement `StageIsolatedNetworkProof` dataclass
    - Implement `ProofArtifactIndex` dataclass
    - Add provenance tracking fields (MATLAB batch, timestamps)
    - Add fingerprint fields for inputs and outputs
    - Add execution metadata fields (timing, resources)
    - Add parity status fields (vertices, edges, strands)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 14.2 Write unit tests for proof artifact data models
    - Test StageIsolatedNetworkProof dataclass
    - Test ProofArtifactIndex dataclass
    - Test serialization/deserialization
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 15. Implement proof artifact generation
  - [ ] 15.1 Add proof generator to source/slavv/parity/proof_artifacts.py
    - Implement `generate_proof_artifact()` function
    - Capture complete provenance chain from network gate execution
    - Generate input/output fingerprints using hashlib
    - Capture execution timing and resource usage
    - Record exact parity status for vertices, edges, and strands
    - Persist proof as JSON to 03_Analysis/proof_artifacts/network_gate_proof_{timestamp}.json
    - Generate human-readable markdown to 03_Analysis/proof_artifacts/network_gate_proof_{timestamp}.md
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.11_

  - [ ]* 15.2 Write unit tests for proof artifact generation
    - Test proof artifact generation from execution metadata
    - Test provenance chain capture
    - Test fingerprint generation
    - Test timing and resource capture
    - Test parity status recording
    - Test JSON and markdown formatting
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.11_

- [ ] 16. Implement proof artifact index maintenance
  - [ ] 16.1 Add index manager to source/slavv/parity/proof_artifacts.py
    - Implement `maintain_proof_artifact_index()` function
    - Maintain chronological list of all proof artifacts
    - Track latest proof reference
    - Track total proof count
    - Persist index to 03_Analysis/proof_artifact_index.json
    - Implement index rebuild from existing artifacts on corruption
    - _Requirements: 4.8, 4.9_

  - [ ]* 16.2 Write unit tests for proof artifact index
    - Test proof artifact index maintenance
    - Test index update with new proof
    - Test latest proof tracking
    - Test index rebuild from existing artifacts
    - _Requirements: 4.8, 4.9_

- [ ] 17. Implement proof artifact CLI commands
  - [ ] 17.1 Add proof display to source/slavv/parity/proof_artifacts.py
    - Implement `display_latest_proof_summary()` function
    - Display proof timestamp
    - Display parity status (vertices/edges/strands)
    - Display execution timing
    - Display MATLAB batch provenance
    - _Requirements: 4.10_

  - [ ] 17.2 Add CLI command for proof artifact display
    - Add `slavv parity-proof --run-dir <path>` command to parity CLI
    - Display latest proof summary
    - List all proof artifacts with timestamps
    - _Requirements: 4.10_

  - [ ]* 17.3 Write integration tests for proof artifact workflow
    - Test proof artifact generation after successful network gate
    - Test proof artifact index update
    - Test proof artifact display command
    - Test proof artifact persistence in both JSON and Markdown
    - _Requirements: 4.1, 4.7, 4.8, 4.10, 4.11_

- [ ] 18. Integrate proof artifacts into network gate workflow
  - [ ] 18.1 Modify source/slavv/parity/network_gate.py for proof generation
    - Add automatic proof artifact generation after successful network gate execution
    - Add proof artifact index update
    - Add proof summary display to CLI output
    - _Requirements: 4.1, 4.7, 4.8_

  - [ ]* 18.2 Write integration tests for proof artifact integration
    - Test automatic proof generation after network gate success
    - Test proof not generated on network gate failure
    - Test proof index maintenance across multiple runs
    - _Requirements: 4.1, 4.8_

- [ ] 19. Phase 4 Checkpoint
  - Ensure all Phase 4 tests pass
  - Verify proof artifacts persist correctly and display properly
  - Ask the user if questions arise

### End-to-End Validation

- [ ] 20. End-to-end workflow validation
  - [ ]* 20.1 Write diagnostic test for complete workflow
    - Test complete workflow from fresh run to reuse eligibility summary
    - Test stage-isolated network gate with imported MATLAB batch
    - Test diagnostic generation after edge parity gap
    - Test proof artifact generation after network parity success
    - Verify all artifacts persist to correct locations (01_Input/, 02_Output/, 03_Analysis/, 99_Metadata/)
    - Verify CLI summaries display correctly at each stage
    - _Requirements: 1.1, 2.5, 3.1, 4.1_

- [ ] 21. Final integration and documentation
  - [ ] 21.1 Verify all modules integrate correctly
    - Verify CLI summaries work with all workflow states
    - Verify network gate integrates with comparison workflow
    - Verify diagnostics integrate with parity gap detection
    - Verify proof artifacts integrate with network gate success
    - _Requirements: 1.1, 2.5, 3.1, 4.1_

  - [ ] 21.2 Run full test suite
    - Run `python -m pytest -m "unit or integration"` for all new tests
    - Run diagnostic tests for end-to-end validation
    - Verify performance targets met (network gate <30s, diagnostics <10s, proof <2s)
    - _Requirements: 2.1_

  - [ ] 21.3 Format and lint all new code
    - Run `python -m ruff format source tests`
    - Run `python -m ruff check source tests --fix`
    - Run `python -m mypy` to verify type checking
    - _Requirements: All_

- [ ] 22. Final Checkpoint
  - Ensure all tests pass
  - Verify all four workflow improvements are complete and functional
  - Ask the user if questions arise

## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at phase boundaries
- The phased approach allows each workflow improvement to be developed and tested independently
- All new modules follow existing repository conventions (pathlib.Path, logging, from __future__ import annotations)
- All artifacts persist to canonical staged layout locations (01_Input/, 02_Output/, 03_Analysis/, 99_Metadata/)
- Performance targets: network gate <30s, diagnostics <10s, proof artifacts <2s
- Backward compatibility maintained throughout all phases
