# Requirements Document

## Introduction

This specification defines the remaining workflow improvements needed to complete the MATLAB vs Python parity workflow tooling. The goal is to finalize four workflow enhancements that will allow developers to efficiently iterate on parity issues while minimizing expensive MATLAB reruns and maximizing the utility of diagnostic information. These improvements focus on CLI messaging clarity, stage-isolated network gate reliability, diagnostic-driven iteration, and proof artifact promotion.

## Glossary

- **CLI**: Command-line interface for the parity comparison tool
- **Parity_Workflow**: The complete MATLAB vs Python comparison workflow system
- **Stage_Isolated_Network_Gate**: A workflow loop that imports exact MATLAB edges and reruns only Python network assembly to isolate parity issues
- **Reuse_Eligibility**: The determination of whether a staged run root can be safely reused for a given workflow loop
- **Loop_Assessment**: The workflow decision logic that determines what operations are safe for a given run root
- **Shared_Neighborhood_Diagnostics**: Diagnostic tools that analyze neighborhood-level claim ordering and partner choice differences between MATLAB and Python
- **Edge_Candidates**: Python module responsible for generating candidate edges during tracing
- **Tracing**: Python module responsible for edge selection and tracing logic
- **Proof_Artifact**: A maintained report or evidence file demonstrating that the stage-isolated network gate produces exact parity
- **MATLAB_Batch**: A completed MATLAB run output that can be imported and reused by Python
- **Python_Checkpoints**: Serialized Python pipeline state files that enable resumption from specific stages

## Requirements

### Requirement 1: CLI Reuse Eligibility Summaries

**User Story:** As a developer, I want clear CLI summaries about reuse eligibility, so that I can quickly understand whether I can reuse a run root, analyze only, or need a fresh MATLAB run.

#### Acceptance Criteria

1. WHEN a comparison run completes successfully, THE CLI SHALL display a reuse eligibility summary
2. THE Reuse_Eligibility_Summary SHALL state whether the run root is safe to reuse for imported-MATLAB parity loops
3. THE Reuse_Eligibility_Summary SHALL state whether the run root is safe for analysis-only comparison
4. THE Reuse_Eligibility_Summary SHALL state whether a fresh MATLAB run is required for the next iteration
5. WHEN the loop assessment determines the run is reusable, THE CLI SHALL display the specific rerun commands that are safe
6. WHEN the loop assessment determines artifacts are missing, THE CLI SHALL explain which artifacts are missing and why they are needed
7. THE Reuse_Eligibility_Summary SHALL include the recommended next action based on the workflow state
8. WHEN multiple workflow loops are available, THE CLI SHALL list all safe reuse options with their corresponding commands

### Requirement 2: Stage-Isolated Network Gate Reliability

**User Story:** As a developer, I want the stage-isolated network gate to remain cheap and reliable, so that I can quickly validate whether parity issues are in edge generation or network assembly without expensive full reruns.

#### Acceptance Criteria

1. WHEN Python reruns from the network stage with imported MATLAB edges, THE Stage_Isolated_Network_Gate SHALL complete in under 30 seconds for the standard test volume
2. THE Stage_Isolated_Network_Gate SHALL verify that all required MATLAB edge artifacts are present before execution
3. IF required MATLAB edge artifacts are missing, THEN THE Stage_Isolated_Network_Gate SHALL fail fast with a descriptive error message
4. THE Stage_Isolated_Network_Gate SHALL force comparison_exact_network mode to ensure deterministic network assembly
5. WHEN the stage-isolated network gate completes, THE CLI SHALL report whether exact parity was achieved at the network stage
6. THE Stage_Isolated_Network_Gate SHALL persist its execution metadata including timing, artifact provenance, and parity status
7. WHEN edge work continues in parallel, THE Stage_Isolated_Network_Gate SHALL remain isolated from edge candidate changes
8. THE Stage_Isolated_Network_Gate SHALL validate input fingerprints match the imported MATLAB batch before execution

### Requirement 3: Shared-Neighborhood Diagnostic Integration

**User Story:** As a developer, I want shared-neighborhood diagnostics to drive the next iteration of edge candidates or tracing logic, so that I can systematically address parity gaps using evidence-based insights.

#### Acceptance Criteria

1. WHEN a parity gap is detected at the edges stage, THE Parity_Workflow SHALL recommend running shared-neighborhood diagnostics
2. THE Shared_Neighborhood_Diagnostics SHALL generate a report identifying neighborhood-level claim ordering differences
3. THE Shared_Neighborhood_Diagnostics SHALL generate a report identifying branch invalidation differences
4. THE Shared_Neighborhood_Diagnostics SHALL generate a report identifying local partner choice differences
5. THE Diagnostic_Report SHALL include specific vertex IDs and neighborhoods where MATLAB and Python diverge
6. THE Diagnostic_Report SHALL quantify candidate coverage differences between MATLAB and Python
7. WHEN diagnostic reports are available, THE CLI SHALL display a summary of the top divergence patterns
8. THE Diagnostic_Report SHALL persist under the 03_Analysis directory in the staged run layout
9. THE Diagnostic_Report SHALL include actionable recommendations for which Edge_Candidates or Tracing code to investigate

### Requirement 4: Proof Artifact Promotion

**User Story:** As a developer, I want a maintained proof artifact for the stage-isolated network gate, so that I can demonstrate and track that network assembly achieves exact parity when given exact MATLAB edges.

#### Acceptance Criteria

1. THE Parity_Workflow SHALL generate a Stage_Isolated_Network_Proof_Report after successful stage-isolated network gate execution
2. THE Stage_Isolated_Network_Proof_Report SHALL document the input MATLAB batch provenance
3. THE Stage_Isolated_Network_Proof_Report SHALL document the exact parity status for vertices, edges, and strands
4. THE Stage_Isolated_Network_Proof_Report SHALL include execution timing and resource usage
5. THE Stage_Isolated_Network_Proof_Report SHALL include checksums or fingerprints of the imported MATLAB edges
6. THE Stage_Isolated_Network_Proof_Report SHALL include checksums or fingerprints of the resulting Python network
7. THE Stage_Isolated_Network_Proof_Report SHALL persist under the 03_Analysis directory with a canonical filename
8. WHEN multiple stage-isolated network gate runs exist, THE Parity_Workflow SHALL maintain a proof artifact index
9. THE Proof_Artifact_Index SHALL list all successful stage-isolated network gate runs with their timestamps and parity status
10. THE CLI SHALL provide a command to display the latest proof artifact summary
11. THE Stage_Isolated_Network_Proof_Report SHALL be formatted as both JSON and human-readable markdown

