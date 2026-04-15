# Design Document: Parity Workflow Completion

## Overview

This design finishes the remaining two backlog items from
`BOTTLENECK_TODO.md` by layering new evidence-producing workflows on top of the
existing parity infrastructure:

1. shared-neighborhood diagnostics for the still-open `edges` parity gap
2. maintained proof artifacts for successful stage-isolated `network`-gate runs

The design assumes the repo already has working reuse-eligibility summaries,
stage-isolated `network`-gate execution, staged run-layout persistence, and
replay-safe normalized comparison params. Those pieces remain the baseline that
the new work plugs into.

This is a design for the next backlog increment, not a description of already
implemented modules. The current codebase already emits a shared-neighborhood
audit during comparison runs and already exposes the stage-isolated network-gate
workflow through the parity CLI; the new work should standardize and preserve
those behaviors while adding the more durable diagnostic and proof surfaces.

The design also treats `dev/reports/tooling/` as archival context. Those
files document a March 23, 2026 snapshot of Ruff, mypy, and pytest findings,
but they are not the live source of truth for current repo health.

## Current Baseline

### Existing Components We Build On

- `source/slavv/parity/workflow_assessment.py`
  - current loop assessment and reuse guidance
- `source/slavv/parity/cli_summaries.py`
  - current CLI reuse-eligibility formatting
- `source/slavv/parity/comparison.py`
  - current comparison orchestration, parity reporting, and network-gate entry
    points
- `source/slavv/parity/metrics.py`
  - current shared-neighborhood audit construction used by comparison runs
- `source/slavv/parity/network_gate.py`
  - current stage-isolated `network`-gate validation and execution
- `source/slavv/parity/run_layout.py`
  - canonical staged artifact layout and path resolution
- `source/slavv/runtime/run_state.py`
  - existing persisted run metadata and compatibility behavior
- `source/slavv/apps/parity_cli.py`
  - current parity CLI entrypoint for comparison and network-gate workflows
- `dev/reports/tooling/README.md`
  - confirms the tooling snapshots are archival reference artifacts

### Design Goal

The active parity diagnosis is no longer "is downstream network assembly
broken?" The backlog says the remaining issue is neighborhood-level claim
ordering, branch invalidation, and local partner choice inside the `edges`
surface. So the design should:

- surface evidence that points investigators at that neighborhood-level gap
- preserve the current `network` gate as the proof that downstream assembly is
  still behaving when fed exact MATLAB edges
- avoid introducing a new heavyweight loop that would slow daily iteration
- avoid turning this workflow-completion spec into a repo-wide cleanup of
  unrelated historical tooling debt

## Architecture

### Component Overview

The remaining work adds two new modules and two small integration layers:

1. **Diagnostics Module** (`source/slavv/parity/diagnostics.py`)
  - analyzes staged MATLAB/Python artifacts for neighborhood-level divergence
  - persists JSON/Markdown reports under `03_Analysis/`
  - emits actionable next-investigation guidance

2. **Diagnostics Integration** (`source/slavv/parity/comparison.py`)
  - recommends diagnostics after an `edges` parity gap
  - displays a compact summary when a diagnostic report already exists
  - upgrades the existing shared-neighborhood audit surface to the canonical
    diagnostic report name without breaking current comparison behavior

3. **Proof Artifact Module** (`source/slavv/parity/proof_artifacts.py`)
   - generates proof artifacts from successful `network`-gate execution data
   - maintains a proof index under `03_Analysis/`
   - formats the latest proof summary for CLI display

4. **Proof Integration** (`source/slavv/parity/network_gate.py` and
   `source/slavv/apps/parity_cli.py`)
   - generates proof artifacts after successful `network`-gate runs
   - exposes `slavv parity-proof --run-dir <path>` for proof inspection

### Data Flow

```text
Comparison run completes
    ->
Existing parity summary and workflow assessment
    ->
[If edges parity gap remains actionable]
    ->
Generate or recommend shared-neighborhood diagnostics
    ->
Persist report to 03_Analysis/
    ->
    Guide next edge_candidates.py / edge_selection.py iteration

Stage-isolated network gate succeeds
    ->
Read existing network-gate execution metadata
    ->
Generate proof artifact + update proof index
    ->
Persist under 03_Analysis/proof_artifacts/
    ->
Expose latest proof summary through CLI
```

## Components and Interfaces

### 1. Shared-Neighborhood Diagnostics

#### Report Models

```python
@dataclass
class NeighborhoodDivergence:
    vertex_id: int | str
    divergence_type: str  # claim_ordering | branch_invalidation | partner_choice
    severity: str  # high | medium | low
    matlab_partner_choices: list[int] = field(default_factory=list)
    python_partner_choices: list[int] = field(default_factory=list)
    matlab_claim_count: int = 0
    python_claim_count: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class SharedNeighborhoodDiagnosticReport:
    run_root: str
    generated_at: str
    matlab_edges_count: int
    python_edges_count: int
    edge_count_delta: int
    claim_ordering_differences: int
    branch_invalidation_differences: int
    partner_choice_differences: int
    divergent_neighborhoods: list[NeighborhoodDivergence]
    matlab_candidate_coverage: dict[str, int]
    python_candidate_coverage: dict[str, int]
    coverage_delta: dict[str, int]
    top_divergence_patterns: list[str]
    recommended_investigations: list[str]
    edge_candidates_to_review: list[str]
    edge_selection_logic_to_review: list[str]
```

#### Generation Interface

```python
def generate_shared_neighborhood_diagnostics(
    run_root: Path,
    *,
    matlab_edges_path: Path | None = None,
    python_edges_path: Path | None = None,
) -> SharedNeighborhoodDiagnosticReport:
    """Generate the canonical shared-neighborhood diagnostic report."""
```

#### Design Notes

- The generator should resolve artifact paths from the staged run root first,
  not require ad hoc paths from the caller unless needed for tests.
- Counts remain important, but the design explicitly treats candidate-coverage
  counts as triage input rather than the final answer.
- Recommendations should point to likely code surfaces, not pretend to identify
  a single exact bug.
- The report should be stable enough to compare across repeated runs on the
  same run root.

#### Persistence Contract

- JSON: `03_Analysis/shared_neighborhood_diagnostics.json`
- Markdown: `03_Analysis/shared_neighborhood_diagnostics.md`

The Markdown report should prioritize:

- top divergence patterns
- a short list of highest-severity neighborhoods
- next code surfaces to inspect in `source/slavv/core/edge_candidates.py` and
  `source/slavv/core/edge_selection.py`

### 2. Comparison-Flow Diagnostics Integration

#### Recommendation Interface

```python
def recommend_diagnostics_if_needed(
    *,
    run_root: Path,
    edges_parity_ok: bool,
    network_gate_parity_ok: bool | None,
) -> str | None:
    """Return a CLI recommendation when diagnostics are the next useful step."""
```

#### Integration Rules

- If `edges` parity succeeded, do not recommend diagnostics.
- If `edges` parity failed and the stage-isolated `network` gate succeeded, the
  recommendation should strongly prefer diagnostics because the remaining gap is
  isolated to the `edges` surface.
- If an existing diagnostic report is already present, `comparison.py` should
  print a compact summary rather than only telling the user to generate one.
- Keep the legacy shared-neighborhood audit artifact readable during the
  transition so older reports and current diagnostics can coexist.
- The integration should not block or slow normal comparison runs when no
  diagnostic report is requested.

### 3. Maintained Proof Artifacts

#### Proof Models

```python
@dataclass
class StageIsolatedNetworkProof:
    run_root: str
    generated_at: str
    matlab_batch_folder: str
    matlab_batch_timestamp: str | None
    matlab_edges_fingerprint: str
    matlab_vertices_fingerprint: str | None
    matlab_energy_fingerprint: str | None
    python_network_fingerprint: str | None
    python_vertices_fingerprint: str | None
    python_edges_fingerprint: str | None
    execution_timestamp: str
    elapsed_seconds: float
    comparison_exact_network_forced: bool
    vertices_exact_parity: bool
    edges_exact_parity: bool
    strands_exact_parity: bool
    overall_parity_achieved: bool
    peak_memory_mb: float | None = None
    cpu_time_seconds: float | None = None


@dataclass
class ProofArtifactIndex:
    run_root: str
    proof_artifacts: list[dict[str, Any]]
    latest_proof: dict[str, Any] | None
    total_proofs: int
```

#### Generation Interface

```python
def generate_proof_artifact(
    execution_metadata_path: Path,
    *,
    run_root: Path,
) -> StageIsolatedNetworkProof:
    """Generate a proof artifact from persisted network-gate execution data."""


def maintain_proof_artifact_index(
    run_root: Path,
    new_proof: StageIsolatedNetworkProof,
) -> ProofArtifactIndex:
    """Update or rebuild the canonical proof index for a run root."""


def display_latest_proof_summary(run_root: Path) -> str:
    """Render the latest proof artifact summary for CLI display."""
```

#### Design Notes

- Proof generation should consume the existing
  `99_Metadata/network_gate_execution.json` artifact rather than re-executing
  any gate logic.
- Resource usage should be optional because the repo runs on Windows and may
  not always have portable metrics available.
- Proof generation should only happen after successful stage-isolated
  `network`-gate runs; failed gates are evidence, but not proof artifacts.
- The proof index is a convenience and durability layer, not the primary source
  of truth; it must be rebuildable from the individual proof files.

#### Persistence Contract

- JSON proofs:
  `03_Analysis/proof_artifacts/network_gate_proof_{timestamp}.json`
- Markdown proofs:
  `03_Analysis/proof_artifacts/network_gate_proof_{timestamp}.md`
- Index:
  `03_Analysis/proof_artifact_index.json`

### 4. CLI Proof Display

#### User Surface

Add:

```text
slavv parity-proof --run-dir <path>
```

This command should:

- load `03_Analysis/proof_artifact_index.json` if it exists
- rebuild from proof files if the index is missing or invalid
- display the latest proof summary
- list known proof timestamps and parity status in a compact form

## Tooling Snapshot Handling

### Archived Snapshot Summary

The archived tooling folder currently records:

- one historical pytest collection failure in
  `dev/tests/diagnostic/test_comparison_setup.py` caused by Python 3.7 evaluating
  built-in generic type syntax
- a small Ruff snapshot covering a mix of test-style issues plus one real
  application error in `source/slavv/apps/web_app.py`
- a large mypy snapshot spanning unrelated modules across the repo

### Design Decision

The diagnostics/proof-artifact work should not absorb that entire backlog.
Instead:

- use the archived reports as awareness for touched files
- keep new modules and tests clean under current repo standards
- narrow tooling debt opportunistically only in files touched by this work
- continue to use the canonical commands from `AGENTS.md` for current
  verification

### Implementation Impact

- New modules `source/slavv/parity/diagnostics.py` and
  `source/slavv/parity/proof_artifacts.py` should be written to pass Ruff and
  mypy cleanly from the start.
- New tests should use modern typing compatible with the repo's supported
  Python version targets for current development, while not relying on the
  archived Python 3.7 collection behavior.
- If comparison or CLI integration touches a file that already appears in the
  archived tooling snapshots, keep edits narrow and avoid expanding unrelated
  lint/type debt.

## Error Handling

### Diagnostics

- Missing required MATLAB or Python artifacts:
  - emit a clear message that diagnostics cannot run yet
  - do not create partial reports unless there is still useful structured
    evidence to persist
- Incompatible or partial edge/candidate artifact content:
  - persist a report with error notes only if the resulting output still helps
    the next investigation
- Recommendation path:
  - if the report cannot be generated, still print the intended next step and
    the artifact that is missing

### Proof Artifacts

- Missing `network_gate_execution.json`:
  - do not create proof artifacts
  - surface a message that a successful stage-isolated `network`-gate run is
    required first
- Index corruption:
  - rebuild from existing proof files
- Proof persistence failure:
  - log a warning and allow the main comparison workflow to continue

## Testing Strategy

### Unit Tests

**Diagnostics** (`dev/tests/unit/parity/test_diagnostics.py`)

- divergence model serialization
- claim-ordering, branch-invalidation, and partner-choice classification
- candidate-coverage delta calculation
- recommendation generation

**Proof Artifacts** (`dev/tests/unit/parity/test_proof_artifacts.py`)

- proof generation from persisted network-gate execution metadata
- optional resource-usage handling
- proof-index updates
- proof-index rebuilds
- latest-proof summary formatting

### Integration Tests

**Diagnostics Integration** (`dev/tests/integration/parity/test_diagnostic_integration.py`)

- recommendation when `edges` parity fails
- stronger recommendation when the `network` gate succeeds
- report persistence to `03_Analysis/`
- CLI summary rendering from a persisted report

**Proof Integration** (`dev/tests/integration/parity/test_proof_artifact_integration.py`)

- proof generation after successful `network`-gate runs
- no proof generation after failed gate runs
- CLI proof display against persisted proof artifacts

### Diagnostic / Workflow Tests

Use at least one workflow-level validation that:

- starts from a run root with an `edges` parity gap
- exercises the diagnostic recommendation/report path
- exercises the successful stage-isolated `network` gate proof path
- verifies artifact placement under the canonical staged layout
- verifies touched-scope tests collect and run under the current environment

## File Changes

### New Modules

- `source/slavv/parity/diagnostics.py`
- `source/slavv/parity/proof_artifacts.py`

### Modified Modules

- `source/slavv/parity/comparison.py`
- `source/slavv/parity/network_gate.py`
- `source/slavv/apps/parity_cli.py`

### Artifact Locations

**Diagnostics**

- `03_Analysis/shared_neighborhood_diagnostics.json`
- `03_Analysis/shared_neighborhood_diagnostics.md`

**Proof Artifacts**

- `03_Analysis/proof_artifacts/network_gate_proof_{timestamp}.json`
- `03_Analysis/proof_artifacts/network_gate_proof_{timestamp}.md`
- `03_Analysis/proof_artifact_index.json`

## Migration Path

### Step 1: Diagnostics First

Build `source/slavv/parity/diagnostics.py`, then wire it into the existing
comparison flow. This closes the first remaining backlog item and improves the
quality of the next `edges`-stage investigation.

### Step 2: Proof Artifacts Second

Build `source/slavv/parity/proof_artifacts.py`, then wire it into successful
stage-isolated `network`-gate runs and the parity CLI. This closes the final
remaining backlog item by turning current network-gate success into a durable,
queryable proof surface.

### Step 3: Final Validation

Run the focused diagnostics/proof tests plus the repo's standard Ruff, mypy,
and pytest checks needed for the touched modules. Do not use the archived
tooling snapshots as a substitute for current validation.

