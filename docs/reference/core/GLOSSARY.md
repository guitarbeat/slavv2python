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
| **Bridge Vertex** | A structural Vertex inserted during Edge Selection. Owned by the Edge Set (bridge fields on the edge payload), not by rewriting the Vertices-stage Vertex Set. |
| **Vertex Set** | Vertices-stage Stage Result: pre-bridge seed set for Edge Discovery/Selection. Network may use Vertex Set ∪ Edge Set bridges as a working list; that composite is not a second Vertices Stage Result. |
| **Edge Set** | The Edges-stage Stage Result: finalized edges after Edge Discovery and Edge Selection. Code: `EdgeSet`. Not a Candidate Set. |

| **Origin** | A starting vertex or seed point from which the extraction pipeline begins searching for edge candidates. (Synonym for Seed Vertex in some contexts) |
| **Edge** | A finalized trace connecting two vertices. Edges represent the local skeleton of the vascular network. Members of an Edge Set. |
| **Edge Discovery** | Identifying connectivity between Vertices from the energy field. Produces a Candidate Set (not an Edge Set). Production strategies: Tracing Discovery (Paper Path) or Watershed Discovery (Exact Route). Not the skimage label-adjacency helper path. |
| **Tracing Discovery** | Paper Path Edge Discovery via directional centerline propagation from Seed Vertices. Code: `TracingDiscovery` (legacy: `MaintainedTracingDiscovery`). |
| **Watershed Discovery** | Exact Route Edge Discovery via regional catchment basins (MATLAB global watershed). Code: `WatershedDiscovery` → `generate_watershed_candidates` (legacy: `FrontierTracingDiscovery`). Exact Route bakes Claimed Trace Energy here at finalize ([ADR 0013](../../adr/0013-claimed-energy-trace-provenance.md)); discovery algorithm otherwise unchanged for the ranking residual. Not skimage `extract_edges_watershed`. |
| **Candidate** | A single potential edge (trace + endpoints/metrics) from Edge Discovery before acceptance into the Edge Set. Not a finalized edge. |
| **Candidate Set** | Authoritative Candidates from Edge Discovery; input to Edge Selection. Mid-stage Edges Checkpoint (not a Stage Result). Code: `CandidateManifest`. Path dual-write (`candidates.pkl` / `checkpoint_edge_candidates.pkl`) is packaging, not two concepts. |
| **Edge Selection** | Post-Edge Discovery: Candidate Set → Edge Set (choose, optional Bridge Vertices, finalize). Ranking uses raw-max ascending `sort_edges` on Candidate energies; Certification ranking parity requires those energies to be Claimed Trace Energy. Sole production path: `select_and_finalize_edge_set`. EdgeManager owns stage lifecycle; residual may call the pure function—must not re-implement selection. No “Chosen Edge Set” domain type. Do not invent cleanup secondary keys that break MATLAB≡Python cleanup on the same Candidate Set. Do not default to a Network rewrite for an Edge Selection Ranking Residual. Closed ranking residual: [Former residual (closed on v18)](EXACT_PROOF_FINDINGS.md#former-residual-closed-on-v18). |
| **Claimed Energy Map** | Watershed-mutated energy volume after claim/penalty writes—the surface MATLAB samples when ranking Candidates during Edge Selection. Distinct from the Energy-stage volume and from original-field samples on Candidate traces. |
| **Claimed Trace Energy** | Candidate/edge energy values sampled from the Claimed Energy Map at Watershed Discovery finalize—the ranking input Edge Selection must see for Certification ranking parity. Hardcoded per-pair overrides and Selection-time claim-map re-sample are not production provenance (re-sample is diagnostic-only). See [ADR 0013](../../adr/0013-claimed-energy-trace-provenance.md). |
| **Edge Selection Ranking Residual** | Certification residual class when discovery emission is otherwise acceptable but Edge Selection keeps/drops wrong undirected pairs because Candidate energies do not reflect the Claimed Energy Map. Closed on the claim root in [ONE TRUTH](EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) via Claimed Trace Energy bake ([ADR 0013](../../adr/0013-claimed-energy-trace-provenance.md)). Mechanism: [Former residual (closed on v18)](EXACT_PROOF_FINDINGS.md#former-residual-closed-on-v18)—do not freeze KPIs here. |
| **Strand** | A connected sequence of one or more edges that represents a distinct vascular branch or segment between junction points. |
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
| **Parity Experiment** 🤖 | Cheap-first same-class compare (raw Candidate Set↔raw, or Edge Set↔Edge Set). Code: `slavv_python.analytics.parity.experiments`. Not Exact Proof Coordinator (Certification) and not Parity Preflight (launch safety). Audit/E-series portfolios are parallel process-integrity guards—not Certification standing or Phase 1 Closure. |
| **Cheap Parity Ladder** 🤖 | Ordered cheap-first sequence (unit → crop → full no-writer) that must be green before accepting a production ranking fix as Certification progress. Guard via `require_cheap_loop`; not itself a Certification claim. |
| **Claim Run Root** 🤖 | Canonical full-volume Parity Run directory named in [ONE TRUTH](EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) as the live Certification claim surface. Historical claim roots stay frozen; after a parity-sensitive fix open a **new** claim root. Diagnostic successors remain non-claim until evaluated ADR 0012 Edges **and** Network pass there. Live root ID only in ONE TRUTH. |
| **Certification** 🤖 | Energy/Vertices: strict discrete + ADR 0011 floats; Edges/Network: ADR 0012 spatial bars—not strict watershed pair-set equality. Phase 1 standing is Certification on the Claim Run Root—not audit/E-series portfolio greens or matlab2python coverage. |
| **Phase 1 Closure** 🤖 | Claimed when a **new** Claim Run Root for full `180709_E` passes **evaluated** ADR 0012 for **both** Edges and Network (after Energy/Vertices pass). Network ADR 0012 pass closes Phase 1; strict-field remains stretch. Definition only—**live whether closed** is [EXACT_PROOF_FINDINGS](EXACT_PROOF_FINDINGS.md); operator brief: [.claude/HANDOFF.md](../../../.claude/HANDOFF.md). |
| **Strict-Field Stretch Goal** 🤖 | Optional strict `connections` / order vs MATLAB (crop KPI), gated on Energy unlock. Distinct from True Zero-Tolerance Stretch. **Network ADR 0012 multiset was the Phase 1 ship gate**, not this stretch. Live whether Phase 1 is closed: [ONE TRUTH](EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk). Historical: ADR 0012 post-v6 addendum. |
| **True Zero-Tolerance Stretch** 🤖 | Post–Phase 1 `--strict-floats` bit-equal program including Energy floats. Default allclose is not stretch success. Live status: findings stretch subsection + dest `stretch_status.json`, not ONE TRUTH. |
| **Stretch Status Taxonomy** 🤖 | `blocked_float_path` / `incomplete_discrete` / `incomplete_infra` / `incomplete_at_full` / `stretch_complete` (`StretchStatus` in `slavv_python.analytics.parity.proof.stretch`). Infra ≠ `blocked_float_path`. Allclose ≠ `stretch_complete`. |
| **Stretch Unlock Token** 🤖 | Crop `stretch_crop_unlock.json` authorizing full stretch for field set `energy` or `energy+discrete`. U5/U6 gated without a matching token. |
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
