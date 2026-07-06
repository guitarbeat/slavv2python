# Glossary

[Up: Documentation Index](../../README.md) · [AGENTS.md](../../../AGENTS.md)

Maintain this reference for domain-specific and project-specific terms used throughout the `slavv2python` repository. This glossary consolidates all terms from both user-facing and AI agent contexts.

> **Glossary Sync Note:**  
> This glossary is **supplementary** to the canonical Domain Glossary in [AGENTS.md § Domain Glossary](../../../AGENTS.md#domain-glossary), which is automatically loaded into AI agent context. This file provides additional technical details and extended definitions. Terms specific to AI agent workflows are marked with 🤖.

---

## Pipeline & Core Concepts

| Term | Definition |
| --- | --- |
| **Pipeline** | The authoritative sequence of computational stages (Energy → Vertices → Edges → Network) required to transform a 3D vascular volume into a vectorized graph representation. |
| **Vertex** | A localized point of interest in the vascular volume, characterized by a 3D position, an estimated radius, and a local energy value. |
| **Seed Vertex** | A Vertex identified directly from the energy field as a local minimum. These serve as the initial discovery points for the Pipeline. |
| **Bridge Vertex** | A structural Vertex inserted during edge selection to resolve overlaps or connectivity gaps. These are topologically necessary but were not originally identified as energy minima. |
| **Vertex Set** | The authoritative collection of Vertices for a given stage of a Run. A Vertex Set can contain both Seed and Bridge vertices. |
| **Origin** | A starting vertex or seed point from which the extraction pipeline begins searching for edge candidates. (Synonym for Seed Vertex in some contexts) |
| **Edge** | A finalized trace connecting two vertices. Edges represent the local skeleton of the vascular network. |
| **Edge Discovery** | The process of identifying potential connectivity between Vertices by analyzing the energy field. |
| **Tracing Discovery** | An Edge Discovery strategy that identifies centerlines via frontier propagation from individual Seed Vertices. |
| **Watershed Discovery** | An Edge Discovery strategy that partitions the volume into regional influence zones (catchment basins) to identify adjacent Vertices. |
| **Strand** | A connected sequence of one or more edges that represents a distinct vascular branch or segment between junction points. |
| **Candidate** | A potential edge trace identified during the frontier-searching or watershed phase. Candidates are evaluated for ownership and cleanup before becoming final edges. |
| **Neighborhood** | The local spatial region around an Origin where multiple origins may compete for candidates. |
| **Frontier** | The active set of pixels at the leading edge of a trace expansion or watershed search. |
| **Lowest Linear Index Priority** | The secondary tie-breaking rule for Vertex and Edge Discovery. When two voxels have identical energy values, the one with the lower Fortran-order linear index is prioritized. |
| **Energy** | A pre-processed image volume (e.g., vesselness, objectness, or Hessian map) that serves as the numerical input for vertex and edge extraction. |

---

## Workflow & Infrastructure

| Term | Definition |
| --- | --- |
| **Staged Run** / **Run State** 🤖 | A structured run directory that follows the canonical `00_Refs/`, `01_Params/`, `02_Output/`, `03_Analysis/`, `99_Metadata/` layout. The complete collection of data persisted during a Run. |
| **Stage Result** 🤖 | The authoritative output of a Pipeline stage, serving as the interface for subsequent stages. |
| **Checkpoint** | Internal state persisted during a stage's execution to allow a Run to recover from interruption or to skip recalculation. Examples: `vertices.pkl`, `edges.pkl`. |
| **Artifact** 🤖 | Supplemental data produced by a stage for diagnostics, auditing, or visualization that is not strictly required for Pipeline progression. |
| **Typed Result Objects** | Structured, validated dataclass models (e.g., `EnergyResult`, `VertexSet`, `EdgeSet`) that serve as the internal and external contract for pipeline stage data. |

---

## Parity & Verification 🤖

| Term | Definition |
| --- | --- |
| **Oracle** | Preserved MATLAB truth vectors and metadata for a specific dataset, stored under `workspace/oracles/`, used as the reference surface for exact parity comparison. |
| **Parity Run** | A disposable developer execution under `workspace/runs/` that compares Python checkpoints against an Oracle via the parity experiment harness. |
| **Parity Preflight** 🤖 | The memory, params-audit, and provenance checks run before a long Parity Run writer starts or resumes. Answers whether it is safe to launch, not whether Python matches MATLAB. |
| **Exact Proof Coordinator** 🤖 | The single orchestration surface that compares Python checkpoints against an Oracle after they exist: `prove-exact`, candidate capture, LUT proof, and edge replay. |
| **Exact Proof** | The process of verifying that Python produces bit-accurate or mathematically equivalent results to the MATLAB oracle. |
| **Parity Experiment** | A structured run that compares a Python pipeline execution against a specific MATLAB oracle, producing matched/missing/extra pair metrics. |
| **Certification** 🤖 | Energy/Vertices: strict discrete + ADR 0011 floats; Edges/Network: ADR 0012 spatial bars—not strict watershed pair-set equality. |
| **Phase 1 Closure** 🤖 | Claimed when canonical full `180709_E` passes per-stage ADR 0012 Edges + Network on `canonical_full_v5` (after Energy/Vertices pass). Operator brief: [.claude/HANDOFF.md](../../../.claude/HANDOFF.md). |
| **Strict-Field Stretch Goal** 🤖 | Optional strict `connections`/strand match vs MATLAB; overlap KPI on `crop_M_exact_v3`; not a Certification blocker. |
| **Canonical Volume** | The single full imaging volume chosen for a Certification milestone. Phase 1 exact-route canonical volume is full `180709_E`. |
| **Parity Pre-Gate** 🤖 | A faster developer loop that exercises the parity harness before Certification on the Canonical Volume. Sequenced as: synthetic smoke, then real crop with its own Oracle, then canonical volume only for the final cert claim. |
| **Synthetic Fixture Volume** | A Python-generated TIFF used for CI and harness smoke tests. Not paired with a preserved MATLAB Oracle unless one is created explicitly for that volume. |
| **Crop Harness Volume** | A real subvolume cut from the `180709` imaging lineage, paired with its own promoted Oracle produced from MATLAB vectorization on that same subvolume. Used for `prove-exact` iteration. |
| **Phase 1 Specification** 🤖 | The single authoritative document for exact-route Certification on full `180709_E`: requirements and implementation together under `docs/plans/phase-1-exact-route-spec.md`. |
| **Exact Proof Findings** 🤖 | The live status log for exact-parity work: active runs, `prove-exact` results, blockers, champion baselines, and a curated index of parity-related compound solutions under `docs/reference/core/EXACT_PROOF_FINDINGS.md`. |

---

## Data Formats

| Term | Definition |
| --- | --- |
| **network.json** | The authoritative versioned JSON export for Python vascular networks, containing schema metadata, validated parameters, vertices, edges, network topology, and optional precomputed summary statistics. |
| **VMV / CASX** | Legacy network export formats still supported for interoperability. |
| **Zarr** | An optional chunked, compressed, N-dimensional array format used for storing large energy volumes during resumable runs. |

---

## Related Documents

- [AGENTS.md](../../../AGENTS.md) — AI agent instructions and full domain glossary
- [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) — System design and component overview
- [EXACT_PROOF_FINDINGS.md](EXACT_PROOF_FINDINGS.md) — Live parity status
- [Documentation Index](../../README.md) — All reference docs
