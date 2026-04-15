# Design Document: Parity Workflow Completion

## Overview

This design completes the MATLAB vs Python parity workflow tooling by implementing four critical workflow improvements: CLI reuse eligibility summaries, stage-isolated network gate reliability, shared-neighborhood diagnostic integration, and proof artifact promotion. These enhancements build on the existing workflow assessment infrastructure (`source/slavv/parity/workflow_assessment.py`) and staged run layout (`source/slavv/parity/run_layout.py`) to provide developers with clear guidance, fast validation loops, evidence-based diagnostics, and maintained proof artifacts.

The design focuses on minimizing expensive MATLAB reruns while maximizing diagnostic utility. The stage-isolated network gate remains the key validation mechanism for isolating parity issues between edge generation and network assembly. The CLI summaries provide actionable next steps based on workflow state, while diagnostic integration drives systematic edge candidate improvements. Proof artifacts maintain evidence that network assembly achieves exact parity when given exact MATLAB edges.

## Architecture

### Component Overview

The implementation extends four existing subsystems:

1. **Workflow Assessment** (`source/slavv/parity/workflow_assessment.py`)
   - Extends `LoopAssessmentReport` with reuse command generation
   - Adds CLI summary formatting for reuse eligibility
   - Implements missing artifact detection and explanation

2. **Stage-Isolated Network Gate** (`source/slavv/parity/comparison.py`)
   - Enhances network-stage validation with timing and fingerprinting
   - Adds fast-fail artifact verification before execution
   - Persists execution metadata and parity status

3. **Diagnostic Integration** (`source/slavv/parity/diagnostics.py` - new module)
   - Implements shared-neighborhood diagnostic report generation
   - Provides CLI integration for diagnostic recommendations
   - Persists reports under `03_Analysis/` with actionable insights

4. **Proof Artifact System** (`source/slavv/parity/proof_artifacts.py` - new module)
   - Generates stage-isolated network proof reports
   - Maintains proof artifact index across multiple runs
   - Provides CLI commands for proof artifact display

### Data Flow

```
User Request
    ↓
Workflow Assessment (assess_loop_request)
    ↓
CLI Summary Generation (format_reuse_eligibility_summary)
    ↓
[If network gate requested]
    ↓
Stage-Isolated Network Gate Execution
    ↓
Proof Artifact Generation
    ↓
[If parity gap detected]
    ↓
Diagnostic Recommendation
    ↓
Shared-Neighborhood Diagnostic Execution
    ↓
Diagnostic Report Persistence
```

### Integration Points

- **Existing**: `LoopAssessmentReport` provides workflow decision surface
- **Existing**: `resolve_run_layout()` provides canonical path resolution
- **Existing**: `RunContext` provides structured run state management
- **New**: CLI summary formatter consumes `LoopAssessmentReport`
- **New**: Network gate validator checks artifact fingerprints
- **New**: Diagnostic module generates neighborhood-level reports
- **New**: Proof artifact system maintains validation evidence

## Components and Interfaces

### 1. CLI Reuse Eligibility Summaries

#### Enhanced LoopAssessmentReport

```python
@dataclass
class LoopAssessmentReport:
    # ... existing fields ...
    
    # New fields for CLI summaries
    reuse_commands: list[str] = field(default_factory=list)
    missing_artifacts: list[str] = field(default_factory=list)
    artifact_explanations: dict[str, str] = field(default_factory=dict)
    next_action_commands: list[str] = field(default_factory=list)
```

#### CLI Summary Formatter

```python
def format_reuse_eligibility_summary(
    report: LoopAssessmentReport,
    *,
    run_root: Path,
    input_file: Path,
) -> str:
    """
    Generate human-readable CLI summary from loop assessment.
    
    Returns formatted text with:
    - Reuse eligibility status
    - Safe workflow loops with specific commands
    - Missing artifacts with explanations
    - Recommended next action
    """
```

#### Command Generator

```python
def generate_reuse_commands(
    report: LoopAssessmentReport,
    *,
    run_root: Path,
    input_file: Path,
) -> list[str]:
    """
    Generate specific rerun commands based on workflow state.
    
    Returns commands for:
    - Analysis-only comparison (--standalone-matlab-dir / --standalone-python-dir)
    - Imported-MATLAB edge rerun (--skip-matlab --python-parity-rerun-from edges)
    - Stage-isolated network gate (--skip-matlab --python-parity-rerun-from network)
    """
```

### 2. Stage-Isolated Network Gate Reliability

#### Network Gate Validator

```python
@dataclass
class NetworkGateValidation:
    """Pre-execution validation for stage-isolated network gate."""
    
    has_matlab_edges: bool
    has_matlab_vertices: bool
    has_matlab_energy: bool
    matlab_edges_fingerprint: str
    matlab_vertices_fingerprint: str
    validation_passed: bool
    validation_errors: list[str]
    validation_timestamp: str
```

```python
def validate_network_gate_artifacts(
    run_root: Path,
) -> NetworkGateValidation:
    """
    Fast-fail validation before network gate execution.
    
    Checks:
    - Required MATLAB edge artifacts present
    - Required MATLAB vertex artifacts present
    - Required MATLAB energy artifacts present
    - Artifact fingerprints for provenance tracking
    
    Returns validation result with specific errors if failed.
    """
```

#### Network Gate Executor

```python
@dataclass
class NetworkGateExecution:
    """Execution metadata for stage-isolated network gate."""
    
    validation: NetworkGateValidation
    started_at: str
    completed_at: str
    elapsed_seconds: float
    comparison_exact_network_forced: bool
    parity_achieved: bool
    vertices_match: bool
    edges_match: bool
    strands_match: bool
    python_network_fingerprint: str
    execution_errors: list[str]
```

```python
def execute_stage_isolated_network_gate(
    run_root: Path,
    *,
    input_file: Path,
    params: dict[str, Any],
) -> NetworkGateExecution:
    """
    Execute stage-isolated network gate with timing and validation.
    
    Process:
    1. Validate required artifacts (fast-fail)
    2. Force comparison_exact_network=True
    3. Import MATLAB edges/vertices/energy
    4. Rerun Python from network stage
    5. Compare results and record parity status
    6. Persist execution metadata
    
    Returns execution result with timing and parity status.
    """
```

### 3. Shared-Neighborhood Diagnostic Integration

#### Diagnostic Report Structure

```python
@dataclass
class NeighborhoodDivergence:
    """Single neighborhood-level divergence."""
    
    vertex_id: int
    matlab_claim_count: int
    python_claim_count: int
    matlab_partner_choices: list[int]
    python_partner_choices: list[int]
    divergence_type: str  # "claim_ordering" | "branch_invalidation" | "partner_choice"
    severity: str  # "high" | "medium" | "low"
```

```python
@dataclass
class SharedNeighborhoodDiagnosticReport:
    """Complete diagnostic report for edge parity gaps."""
    
    run_root: str
    generated_at: str
    matlab_edges_count: int
    python_edges_count: int
    edge_count_delta: int
    
    # Neighborhood-level analysis
    divergent_neighborhoods: list[NeighborhoodDivergence]
    claim_ordering_differences: int
    branch_invalidation_differences: int
    partner_choice_differences: int
    
    # Candidate coverage analysis
    matlab_candidate_coverage: dict[str, int]
    python_candidate_coverage: dict[str, int]
    coverage_delta: dict[str, int]
    
    # Actionable recommendations
    top_divergence_patterns: list[str]
    recommended_investigations: list[str]
    edge_candidates_to_review: list[str]
    tracing_logic_to_review: list[str]
```

#### Diagnostic Generator

```python
def generate_shared_neighborhood_diagnostics(
    run_root: Path,
    *,
    matlab_edges_path: Path,
    python_edges_path: Path,
) -> SharedNeighborhoodDiagnosticReport:
    """
    Generate neighborhood-level diagnostic report.
    
    Analysis:
    - Compare claim ordering per neighborhood
    - Identify branch invalidation differences
    - Detect partner choice divergences
    - Quantify candidate coverage gaps
    - Generate actionable recommendations
    
    Returns complete diagnostic report.
    """
```

#### CLI Integration

```python
def recommend_diagnostics_if_needed(
    parity_status: dict[str, bool],
    run_root: Path,
) -> str | None:
    """
    Recommend diagnostic execution when parity gaps detected.
    
    Returns CLI message with diagnostic command if:
    - Edges parity failed
    - Network parity succeeded (isolates issue to edges)
    """
```

### 4. Proof Artifact Promotion

#### Proof Report Structure

```python
@dataclass
class StageIsolatedNetworkProof:
    """Proof artifact for stage-isolated network gate."""
    
    # Provenance
    run_root: str
    generated_at: str
    matlab_batch_folder: str
    matlab_batch_timestamp: str
    
    # Input fingerprints
    matlab_edges_fingerprint: str
    matlab_vertices_fingerprint: str
    matlab_energy_fingerprint: str
    
    # Execution details
    execution_timestamp: str
    elapsed_seconds: float
    comparison_exact_network_forced: bool
    
    # Parity status
    vertices_exact_parity: bool
    edges_exact_parity: bool
    strands_exact_parity: bool
    overall_parity_achieved: bool
    
    # Output fingerprints
    python_network_fingerprint: str
    python_vertices_fingerprint: str
    python_edges_fingerprint: str
    
    # Resource usage
    peak_memory_mb: float | None
    cpu_time_seconds: float | None
```

#### Proof Artifact Manager

```python
def generate_proof_artifact(
    execution: NetworkGateExecution,
    *,
    run_root: Path,
    matlab_batch_folder: Path,
) -> StageIsolatedNetworkProof:
    """
    Generate proof artifact from network gate execution.
    
    Captures:
    - Complete provenance chain
    - Input/output fingerprints
    - Execution timing and resources
    - Exact parity status
    
    Persists as both JSON and Markdown.
    """
```

```python
@dataclass
class ProofArtifactIndex:
    """Index of all proof artifacts for a run root."""
    
    run_root: str
    proof_artifacts: list[dict[str, Any]]  # Sorted by timestamp
    latest_proof: dict[str, Any] | None
    total_proofs: int
```

```python
def maintain_proof_artifact_index(
    run_root: Path,
    new_proof: StageIsolatedNetworkProof,
) -> ProofArtifactIndex:
    """
    Update proof artifact index with new proof.
    
    Maintains:
    - Chronological list of all proofs
    - Latest proof reference
    - Proof count
    
    Persists index to 03_Analysis/proof_artifact_index.json
    """
```

#### CLI Commands

```python
def display_latest_proof_summary(run_root: Path) -> str:
    """
    Display summary of latest proof artifact.
    
    Shows:
    - Proof timestamp
    - Parity status (vertices/edges/strands)
    - Execution timing
    - MATLAB batch provenance
    """
```

## Data Models

### Enhanced Workflow Assessment

```python
# Extension to existing LoopAssessmentReport
reuse_commands: list[str]
# Example: [
#     "python workspace/scripts/cli/compare_matlab_python.py --skip-matlab --resume-latest --python-parity-rerun-from edges",
#     "python workspace/scripts/cli/compare_matlab_python.py --skip-matlab --resume-latest --python-parity-rerun-from network"
# ]

missing_artifacts: list[str]
# Example: ["MATLAB edges checkpoint", "Python network.json"]

artifact_explanations: dict[str, str]
# Example: {
#     "MATLAB edges checkpoint": "Required for stage-isolated network gate validation",
#     "Python network.json": "Required for analysis-only comparison"
# }

next_action_commands: list[str]
# Example: ["python workspace/scripts/cli/compare_matlab_python.py --input data/slavv_test_volume.tif --output-dir comparison_output"]
```

### Network Gate Validation

```python
{
    "has_matlab_edges": true,
    "has_matlab_vertices": true,
    "has_matlab_energy": true,
    "matlab_edges_fingerprint": "sha256:abc123...",
    "matlab_vertices_fingerprint": "sha256:def456...",
    "validation_passed": true,
    "validation_errors": [],
    "validation_timestamp": "2026-04-15T10:30:00Z"
}
```

### Diagnostic Report

```python
{
    "run_root": "/path/to/comparison_output",
    "generated_at": "2026-04-15T10:35:00Z",
    "matlab_edges_count": 1234,
    "python_edges_count": 1198,
    "edge_count_delta": -36,
    "divergent_neighborhoods": [
        {
            "vertex_id": 42,
            "matlab_claim_count": 8,
            "python_claim_count": 6,
            "matlab_partner_choices": [10, 15, 23],
            "python_partner_choices": [10, 15],
            "divergence_type": "partner_choice",
            "severity": "high"
        }
    ],
    "claim_ordering_differences": 12,
    "branch_invalidation_differences": 5,
    "partner_choice_differences": 19,
    "matlab_candidate_coverage": {
        "forward_trace": 856,
        "backward_trace": 378
    },
    "python_candidate_coverage": {
        "forward_trace": 820,
        "backward_trace": 378
    },
    "coverage_delta": {
        "forward_trace": -36,
        "backward_trace": 0
    },
    "top_divergence_patterns": [
        "Partner choice divergence in high-degree vertices (19 cases)",
        "Claim ordering differences in branching neighborhoods (12 cases)"
    ],
    "recommended_investigations": [
        "Review edge_candidates.py forward trace logic for high-degree vertices",
        "Investigate tracing.py claim ordering in branching scenarios"
    ],
    "edge_candidates_to_review": [
        "source/slavv/processing/edge_candidates.py:forward_trace_candidates"
    ],
    "tracing_logic_to_review": [
        "source/slavv/processing/tracing.py:select_best_partner"
    ]
}
```

### Proof Artifact

```python
{
    "run_root": "/path/to/comparison_output",
    "generated_at": "2026-04-15T10:40:00Z",
    "matlab_batch_folder": "01_Input/matlab_results/batch_260415_103000",
    "matlab_batch_timestamp": "2026-04-15T10:30:00Z",
    "matlab_edges_fingerprint": "sha256:abc123...",
    "matlab_vertices_fingerprint": "sha256:def456...",
    "matlab_energy_fingerprint": "sha256:ghi789...",
    "execution_timestamp": "2026-04-15T10:40:00Z",
    "elapsed_seconds": 18.5,
    "comparison_exact_network_forced": true,
    "vertices_exact_parity": true,
    "edges_exact_parity": true,
    "strands_exact_parity": true,
    "overall_parity_achieved": true,
    "python_network_fingerprint": "sha256:jkl012...",
    "python_vertices_fingerprint": "sha256:mno345...",
    "python_edges_fingerprint": "sha256:pqr678...",
    "peak_memory_mb": 512.3,
    "cpu_time_seconds": 17.2
}
```

## Error Handling

### CLI Summary Generation

**Error**: Loop assessment report missing or incomplete
- **Handling**: Generate minimal summary with warning
- **User Message**: "Workflow assessment incomplete. Run with --validate-only to refresh."

**Error**: Run root not found or inaccessible
- **Handling**: Return error message with path resolution guidance
- **User Message**: "Run root not found: {path}. Verify path and permissions."

### Network Gate Validation

**Error**: Required MATLAB artifacts missing
- **Handling**: Fast-fail before execution with specific artifact list
- **User Message**: "Stage-isolated network gate blocked. Missing artifacts: {list}. Run imported-MATLAB edge rerun first."

**Error**: Artifact fingerprint mismatch
- **Handling**: Warn about potential stale artifacts
- **User Message**: "Warning: MATLAB artifact fingerprints changed since last validation. Results may not be comparable."

**Error**: Network gate execution timeout (>60s)
- **Handling**: Terminate execution and log timing issue
- **User Message**: "Network gate exceeded 60s timeout. Investigate performance regression."

### Diagnostic Generation

**Error**: MATLAB or Python edges not found
- **Handling**: Return error with artifact location guidance
- **User Message**: "Cannot generate diagnostics. Missing edge artifacts at: {paths}"

**Error**: Edge format incompatible
- **Handling**: Log format version and skip incompatible analysis
- **User Message**: "Edge format version mismatch. Diagnostics may be incomplete."

**Error**: Diagnostic computation failure
- **Handling**: Persist partial results with error annotation
- **User Message**: "Diagnostic generation partially failed: {error}. Partial results saved."

### Proof Artifact Generation

**Error**: Network gate execution failed
- **Handling**: Do not generate proof artifact
- **User Message**: "Proof artifact not generated due to execution failure."

**Error**: Proof artifact persistence failure
- **Handling**: Log error but continue workflow
- **User Message**: "Warning: Could not persist proof artifact: {error}"

**Error**: Proof index corruption
- **Handling**: Rebuild index from existing proof artifacts
- **User Message**: "Proof index rebuilt from {count} existing artifacts."

## Testing Strategy

This feature involves workflow orchestration, file I/O, subprocess execution, and CLI integration. Testing will use a combination of unit tests for pure logic, integration tests for workflow coordination, and diagnostic tests for end-to-end validation.

### Unit Tests

**CLI Summary Formatting** (`tests/unit/parity/test_cli_summaries.py`)
- Test `format_reuse_eligibility_summary()` with various loop assessment states
- Test `generate_reuse_commands()` command generation for each workflow loop
- Test missing artifact explanation formatting
- Test next action command generation

**Network Gate Validation** (`tests/unit/parity/test_network_gate_validation.py`)
- Test `validate_network_gate_artifacts()` with complete artifacts
- Test validation failure with missing MATLAB edges
- Test validation failure with missing MATLAB vertices
- Test fingerprint generation for artifacts
- Test validation error message formatting

**Diagnostic Report Generation** (`tests/unit/parity/test_diagnostic_generation.py`)
- Test neighborhood divergence detection logic
- Test candidate coverage delta calculation
- Test recommendation generation from divergence patterns
- Test diagnostic report serialization

**Proof Artifact Management** (`tests/unit/parity/test_proof_artifacts.py`)
- Test proof artifact generation from execution metadata
- Test proof artifact index maintenance
- Test proof artifact index rebuild from existing artifacts
- Test latest proof summary formatting

### Integration Tests

**CLI Reuse Eligibility Flow** (`tests/integration/parity/test_reuse_eligibility_integration.py`)
- Test complete flow from loop assessment to CLI summary display
- Test reuse command generation with actual run root
- Test missing artifact detection with partial run state
- Test next action recommendation with various workflow states

**Stage-Isolated Network Gate Flow** (`tests/integration/parity/test_network_gate_integration.py`)
- Test complete network gate execution with imported MATLAB artifacts
- Test fast-fail validation with missing artifacts
- Test timing measurement and 30s performance target
- Test parity status reporting
- Test execution metadata persistence

**Diagnostic Integration Flow** (`tests/integration/parity/test_diagnostic_integration.py`)
- Test diagnostic recommendation after parity gap detection
- Test diagnostic report generation with real edge artifacts
- Test diagnostic report persistence to 03_Analysis/
- Test CLI diagnostic summary display

**Proof Artifact Flow** (`tests/integration/parity/test_proof_artifact_integration.py`)
- Test proof artifact generation after successful network gate
- Test proof artifact index update
- Test proof artifact display command
- Test proof artifact persistence in both JSON and Markdown

### Diagnostic Tests

**End-to-End Workflow** (`tests/diagnostic/test_parity_workflow_completion.py`)
- Test complete workflow from fresh run to reuse eligibility summary
- Test stage-isolated network gate with imported MATLAB batch
- Test diagnostic generation after edge parity gap
- Test proof artifact generation after network parity success
- Verify all artifacts persist to correct locations
- Verify CLI summaries display correctly

**Performance Validation** (`tests/diagnostic/test_network_gate_performance.py`)
- Test network gate completes in <30s for standard test volume
- Test validation overhead is <1s
- Test proof artifact generation overhead is <2s

### Test Data Requirements

- **Staged run root** with complete MATLAB batch and Python checkpoints
- **Partial run root** with missing artifacts for validation testing
- **Edge parity gap scenario** with known divergences for diagnostic testing
- **Network parity success scenario** for proof artifact testing
- **Standard test volume** (`data/slavv_test_volume.tif`) for performance testing

### Test Execution

```powershell
# Unit tests
python -m pytest tests/unit/parity/test_cli_summaries.py
python -m pytest tests/unit/parity/test_network_gate_validation.py
python -m pytest tests/unit/parity/test_diagnostic_generation.py
python -m pytest tests/unit/parity/test_proof_artifacts.py

# Integration tests
python -m pytest tests/integration/parity/test_reuse_eligibility_integration.py
python -m pytest tests/integration/parity/test_network_gate_integration.py
python -m pytest tests/integration/parity/test_diagnostic_integration.py
python -m pytest tests/integration/parity/test_proof_artifact_integration.py

# Diagnostic tests
python -m pytest tests/diagnostic/test_parity_workflow_completion.py
python -m pytest tests/diagnostic/test_network_gate_performance.py

# Full parity test suite
python -m pytest -m "unit or integration" tests/unit/parity/ tests/integration/parity/
```

## Implementation Notes

### File Locations

**New Modules**:
- `source/slavv/parity/cli_summaries.py` - CLI summary formatting
- `source/slavv/parity/network_gate.py` - Network gate validation and execution
- `source/slavv/parity/diagnostics.py` - Diagnostic report generation
- `source/slavv/parity/proof_artifacts.py` - Proof artifact management

**Modified Modules**:
- `source/slavv/parity/workflow_assessment.py` - Enhanced `LoopAssessmentReport`
- `source/slavv/parity/comparison.py` - Network gate integration
- `source/slavv/apps/parity_cli.py` - CLI summary display

**New Test Files**:
- `tests/unit/parity/test_cli_summaries.py`
- `tests/unit/parity/test_network_gate_validation.py`
- `tests/unit/parity/test_diagnostic_generation.py`
- `tests/unit/parity/test_proof_artifacts.py`
- `tests/integration/parity/test_reuse_eligibility_integration.py`
- `tests/integration/parity/test_network_gate_integration.py`
- `tests/integration/parity/test_diagnostic_integration.py`
- `tests/integration/parity/test_proof_artifact_integration.py`
- `tests/diagnostic/test_parity_workflow_completion.py`
- `tests/diagnostic/test_network_gate_performance.py`

### Artifact Persistence Locations

Following the canonical staged layout from `docs/reference/COMPARISON_LAYOUT.md`:

**Workflow Assessment**:
- `99_Metadata/loop_assessment.json` (existing, enhanced)

**Network Gate**:
- `99_Metadata/network_gate_validation.json` (new)
- `99_Metadata/network_gate_execution.json` (new)

**Diagnostics**:
- `03_Analysis/shared_neighborhood_diagnostics.json` (new)
- `03_Analysis/shared_neighborhood_diagnostics.md` (new, human-readable)

**Proof Artifacts**:
- `03_Analysis/proof_artifacts/network_gate_proof_{timestamp}.json` (new)
- `03_Analysis/proof_artifacts/network_gate_proof_{timestamp}.md` (new)
- `03_Analysis/proof_artifact_index.json` (new)

### CLI Integration Points

**Reuse Eligibility Summary**:
- Display after successful comparison run
- Display after `--validate-only` run
- Display after `--resume-latest` compatibility check

**Network Gate Execution**:
- Triggered by `--python-parity-rerun-from network`
- Automatic validation before execution
- Automatic proof artifact generation after success

**Diagnostic Recommendation**:
- Display after edge parity gap detection
- Provide specific diagnostic command
- Link to diagnostic report location

**Proof Artifact Display**:
- New CLI command: `slavv parity-proof --run-dir <path>`
- Display latest proof summary
- List all proof artifacts with timestamps

### Performance Targets

**Network Gate Execution**: <30s for standard test volume
- Validation overhead: <1s
- Python network rerun: <25s
- Comparison and reporting: <4s

**Diagnostic Generation**: <10s for standard test volume
- Edge loading: <2s
- Neighborhood analysis: <6s
- Report generation: <2s

**Proof Artifact Generation**: <2s
- Fingerprint computation: <1s
- Report formatting: <1s

### Backward Compatibility

All enhancements maintain backward compatibility:
- Existing `LoopAssessmentReport` fields unchanged
- New fields have default values
- Existing CLI commands unchanged
- New CLI summaries are additive
- Existing artifact locations preserved
- New artifacts use new locations

### Dependencies

**Existing**:
- `source/slavv/parity/workflow_assessment.py`
- `source/slavv/parity/run_layout.py`
- `source/slavv/parity/comparison.py`
- `source/slavv/runtime/run_state.py`

**New**:
- No new external dependencies required
- Uses existing `hashlib` for fingerprinting
- Uses existing `json` for persistence
- Uses existing `pathlib` for file operations

## Migration Path

### Phase 1: CLI Summaries (Requirement 1)

1. Extend `LoopAssessmentReport` with new fields
2. Implement `format_reuse_eligibility_summary()`
3. Implement `generate_reuse_commands()`
4. Integrate into `parity_cli.py` display logic
5. Add unit tests for summary formatting
6. Add integration tests for CLI display

### Phase 2: Network Gate Reliability (Requirement 2)

1. Implement `validate_network_gate_artifacts()`
2. Implement `execute_stage_isolated_network_gate()`
3. Integrate validation into comparison workflow
4. Add timing measurement and reporting
5. Add unit tests for validation logic
6. Add integration tests for execution flow
7. Add performance diagnostic tests

### Phase 3: Diagnostic Integration (Requirement 3)

1. Implement `SharedNeighborhoodDiagnosticReport` structure
2. Implement `generate_shared_neighborhood_diagnostics()`
3. Implement `recommend_diagnostics_if_needed()`
4. Integrate into comparison workflow
5. Add unit tests for diagnostic generation
6. Add integration tests for CLI integration
7. Add diagnostic tests for end-to-end flow

### Phase 4: Proof Artifacts (Requirement 4)

1. Implement `StageIsolatedNetworkProof` structure
2. Implement `generate_proof_artifact()`
3. Implement `maintain_proof_artifact_index()`
4. Implement `display_latest_proof_summary()`
5. Add CLI command for proof display
6. Add unit tests for proof management
7. Add integration tests for proof generation
8. Add diagnostic tests for proof workflow

Each phase can be developed and tested independently, with integration occurring at the end of each phase.
