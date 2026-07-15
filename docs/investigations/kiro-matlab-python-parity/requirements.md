# Requirements Document

## Introduction

This feature delivers and certifies 100% MATLAB-to-Python parity for the SLAVV
(Segmentation-Less, Automated, Vascular Vectorization) pipeline. The goal is a full Python
port of the released SLAVV MATLAB source (`external/Vectorization-Public/source/`) whose
output is proven equivalent to preserved MATLAB truth vectors across all four pipeline
stages: Energy → Vertices → Edges → Network.

"100% parity" is defined operationally as **Certification**: sequential exact-parity gates
report zero missing and zero extra on discrete and topological fields, and `np.allclose`
agreement on continuous floating-point fields, for every required pipeline stage on the
defined Canonical Volume. Because bit-exact floating-point equality is unachievable across
MATLAB and NumPy math libraries (floating-point non-associativity plus BLAS/FFT/ISA
non-determinism), and because the watershed is an order-sensitive shared-state flood-fill,
parity is certified through tolerance-based and spatial/topological bars rather than raw
output-set equality. This approach follows the golden-master/oracle testing methodology
recorded in `PARITY_METHODOLOGY.md` and codified in ADR 0011 and ADR 0012.

The released MATLAB source is the executable specification. Preserved MATLAB vectors are
the proof oracle. This document states what the Python port and its certification harness
must do, not how to implement them.

## Glossary

- **SLAVV_Pipeline**: The authoritative sequence of computational stages (Energy → Vertices → Edges → Network) that transforms a 3D vascular volume into a vectorized graph.
- **Energy_Stage**: The pipeline stage that produces the multi-scale energy field, per-voxel best scale indices, and derived size image from raw image intensities.
- **Vertex_Stage**: The pipeline stage that extracts vertices (position, scale/radius, local energy) from the energy field.
- **Edge_Stage**: The pipeline stage that discovers edges connecting vertices via the watershed flood-fill of the energy field.
- **Network_Stage**: The pipeline stage that assembles edges into strands and identifies bifurcations to form the final network graph.
- **Exact_Route**: The memory-safe, `float64`, `[Y, X, Z]` Fortran-order Python workflow, activated by `comparison_exact_network`, that targets bit-faithful MATLAB parity using the incremental octave-chunked energy engine.
- **Parity_Harness**: The `slavv parity` tooling (`prove-exact`, `prove-exact-sequence`) that compares Python checkpoints against an Oracle and emits parity reports.
- **Oracle**: Preserved MATLAB truth vectors and metadata for a specific dataset, stored under `workspace/oracles/`, serving as the reference surface for parity comparison.
- **Oracle_Loader**: The component that reads preserved MATLAB `.mat` / HDF5 artifacts into Python comparison surfaces, applying index-base and axis-order conventions.
- **Canonical_Volume**: The single full imaging volume chosen for a Certification milestone. The Phase 1 Canonical Volume is the full `180709_E` volume.
- **Certification_Report**: The structured output (JSON) produced by the Parity_Harness for a stage, recording missing/extra counts, float agreement, and pass/fail verdict.
- **Discrete_Field**: A parity field whose values are counts, indices, positions, connections, or topological identifiers, compared for strict equality.
- **Continuous_Field**: A parity field whose values are floating-point quantities (for example `energy.energy`, `lumen_radius_microns`), compared within a numerical tolerance.
- **Float_Tolerance**: The `np.allclose` criterion `|actual − desired| ≤ atol + rtol·|desired|` with `rtol = 1e-7` and `atol = 1e-9`, per ADR 0011.
- **Ownership_Map**: The per-voxel `vertex_index_map` assigning each voxel to a winning vertex catchment basin, used as the Edge_Stage spatial parity bar.
- **Trace_Tolerance**: A sub-voxel maximum-distance bound used to compare per-edge and per-strand geometric traces, per ADR 0012.
- **Endpoint_Pair_Multiset**: The order-independent multiset of strand endpoint vertex pairs, used as a Network_Stage topological parity bar.
- **Bifurcation_Multiset**: The order-independent multiset of bifurcation points, used as a Network_Stage topological parity bar.
- **Certification**: The state in which sequential parity gates report zero missing and zero extra on Discrete_Fields, `np.allclose` agreement on Continuous_Fields, and the ADR 0012 spatial/topological bars for Edge_Stage and Network_Stage, for every required stage on the Canonical_Volume.

## Requirements

### Requirement 1: Full Native Python Port of the SLAVV Pipeline

**User Story:** As a maintainer, I want the complete SLAVV method implemented as native Python code, so that the four-stage pipeline runs from TIFF to network without runtime dependence on the MATLAB runtime or imported MATLAB energy artifacts.

#### Acceptance Criteria

1. THE SLAVV_Pipeline SHALL execute the Energy_Stage, Vertex_Stage, Edge_Stage, and Network_Stage in Python from a 3D input volume to a network graph.
2. WHEN the Exact_Route is enabled through `comparison_exact_network`, THE SLAVV_Pipeline SHALL discover vertices natively in Python without injecting Oracle vertex checkpoints.
3. THE Energy_Stage SHALL derive the energy field from raw image intensities using the native Python Hessian matched-filtering backend identified as `python_native_hessian`.
4. WHERE the released MATLAB source and the Python port differ in algorithm structure, THE SLAVV_Pipeline SHALL reproduce the MATLAB mathematical method and algorithm structure documented in `external/Vectorization-Public/source/`.

### Requirement 2: Mandatory float64 Precision

**User Story:** As a parity engineer, I want all continuous quantities computed in double precision, so that Python accumulation does not introduce precision loss beyond documented cross-library floating-point drift.

#### Acceptance Criteria

1. THE Energy_Stage SHALL compute energies, coordinates, and derived scale quantities using `float64`.
2. THE Vertex_Stage SHALL compute positions, scales, and energies using `float64`.
3. THE Edge_Stage SHALL compute coordinates, energies, distance penalties, and suppression factors using `float64`.
4. THE Network_Stage SHALL compute strand geometry coordinates using `float64`.

### Requirement 3: MATLAB Grid Alignment and Column-Major Tie-Breaking

**User Story:** As a parity engineer, I want the Exact_Route to use MATLAB's memory layout and tie-breaking, so that discrete selection outcomes match MATLAB's column-major ordering.

#### Acceptance Criteria

1. THE Exact_Route SHALL represent voxel volumes using the internal `[Y, X, Z]` grid orientation with Fortran memory order.
2. WHEN two candidate voxels have equal energy values, THE SLAVV_Pipeline SHALL select the voxel with the lower Fortran-order linear index.
3. WHEN converting a persisted artifact to physical storage, THE SLAVV_Pipeline SHALL map the internal `[Y, X, Z]` orientation to the physical `[Z, Y, X]` orientation.
4. WHEN rounding a coordinate at a `.5` boundary during painting or candidate filtering, THE SLAVV_Pipeline SHALL round half away from zero.

### Requirement 4: Energy Stage Parity

**User Story:** As a parity engineer, I want the Energy_Stage certified against the Oracle, so that per-voxel scale selection is exact and energy values agree within tolerance.

#### Acceptance Criteria

1. WHEN the Parity_Harness proves the Energy_Stage against the Oracle, THE Parity_Harness SHALL report zero missing and zero extra for the `scale_indices` Discrete_Field.
2. WHEN the Parity_Harness proves the Energy_Stage against the Oracle, THE Parity_Harness SHALL certify the `energy.energy` Continuous_Field within the Float_Tolerance.
3. WHEN the Parity_Harness proves the Energy_Stage against the Oracle, THE Parity_Harness SHALL certify the `lumen_radius_microns` Continuous_Field within the Float_Tolerance.
4. THE Energy_Stage SHALL construct the coarse-to-fine interpolation mesh using MATLAB-equivalent `linspace` endpoints so that per-voxel scale selection matches MATLAB at coarse-cell boundaries.

### Requirement 5: Vertex Stage Parity

**User Story:** As a parity engineer, I want the Vertex_Stage certified against the Oracle, so that vertex positions and scales match MATLAB exactly and energies agree within tolerance.

#### Acceptance Criteria

1. WHEN the Parity_Harness proves the Vertex_Stage against the Oracle, THE Parity_Harness SHALL report zero missing and zero extra for vertex positions.
2. WHEN the Parity_Harness proves the Vertex_Stage against the Oracle, THE Parity_Harness SHALL report zero missing and zero extra for vertex scales.
3. WHEN the Parity_Harness proves the Vertex_Stage against the Oracle, THE Parity_Harness SHALL certify vertex energies within the Float_Tolerance.
4. THE Vertex_Stage SHALL compute structuring-element voxel membership using MATLAB float-radius membership equivalent to `construct_structuring_element.m`.

### Requirement 6: Edge Stage Parity via Spatial Ownership Map

**User Story:** As a parity engineer, I want the Edge_Stage certified on the voxel ownership map and trace tolerance, so that parity reflects the faithful watershed math rather than the chaotic emission order.

#### Acceptance Criteria

1. WHEN the Parity_Harness proves the Edge_Stage against the Oracle, THE Parity_Harness SHALL compare the Python Ownership_Map against the MATLAB Ownership_Map as the edge spatial parity bar.
2. WHEN the Parity_Harness proves the Edge_Stage against the Oracle, THE Parity_Harness SHALL certify per-edge geometric traces within the Trace_Tolerance.
3. THE Parity_Harness SHALL exclude raw edge-pair overlap from Edge_Stage certification metrics.
4. THE Edge_Stage SHALL keep discrete watershed inputs bit-faithful to MATLAB, including orientation, lookup tables, structuring-element offsets, and `edge_number_tolerance`.
5. WHERE the Exact_Route is enabled, THE Edge_Stage SHALL disable conflict painting to preserve MATLAB watershed faithfulness.

### Requirement 7: Network Stage Parity via Topology Multisets and Trace Tolerance

**User Story:** As a parity engineer, I want the Network_Stage certified on topology multisets and sub-voxel geometry, so that strand and bifurcation structure reproduce MATLAB order-independently.

#### Acceptance Criteria

1. WHEN the Parity_Harness proves the Network_Stage against the Oracle, THE Parity_Harness SHALL report zero missing and zero extra for the Endpoint_Pair_Multiset.
2. WHEN the Parity_Harness proves the Network_Stage against the Oracle, THE Parity_Harness SHALL report zero missing and zero extra for the Bifurcation_Multiset.
3. WHEN the Parity_Harness proves the Network_Stage against the Oracle, THE Parity_Harness SHALL certify per-strand geometric traces within the Trace_Tolerance.
4. THE Parity_Harness SHALL exclude raw edge-pair overlap from Network_Stage certification metrics.

### Requirement 8: Sequential Stage Gating

**User Story:** As a parity engineer, I want stages certified in dependency order, so that an upstream failure localizes the divergence and blocks downstream claims.

#### Acceptance Criteria

1. THE Parity_Harness SHALL evaluate stages in the order Energy_Stage, then Vertex_Stage, then Edge_Stage, then Network_Stage.
2. IF a stage fails its parity bar, THEN THE Parity_Harness SHALL block Certification of that stage and all downstream stages for the workflow under test.
3. WHILE proving a stage, THE Parity_Harness SHALL use the same destination run root for the stage under test and its upstream stages.
4. THE Parity_Harness SHALL identify the first failing parity field in the Certification_Report when a stage fails.

### Requirement 9: Certification on the Canonical Volume

**User Story:** As a maintainer, I want a reproducible Certification on a single full volume, so that the parity claim rests on the canonical dataset rather than informal match-rate milestones.

#### Acceptance Criteria

1. THE Certification milestone SHALL use the full `180709_E` volume as the Canonical_Volume.
2. WHEN all four stages pass their parity bars sequentially on the Canonical_Volume, THE Parity_Harness SHALL report the Exact_Route as certified for the Canonical_Volume.
3. THE Parity_Harness SHALL derive Certification only from a native Exact_Route run on the Canonical_Volume.

### Requirement 10: Oracle Artifact Management

**User Story:** As a parity engineer, I want preserved MATLAB truth stored with one loadable artifact per gated stage, so that every stage has a reference surface for comparison.

#### Acceptance Criteria

1. THE Oracle SHALL store preserved MATLAB truth for the Canonical_Volume under `workspace/oracles/180709_E_full_v2`.
2. THE Oracle SHALL provide one loadable artifact for each gated stage: Energy_Stage, Vertex_Stage, Edge_Stage, and Network_Stage.
3. WHEN the Oracle_Loader reads a MATLAB scale-index artifact, THE Oracle_Loader SHALL apply exactly one one-based-to-zero-based index shift.
4. WHEN the Oracle_Loader reads a v7.3 HDF5 artifact, THE Oracle_Loader SHALL apply the reversed-axis convention to align with the internal orientation.

### Requirement 11: Oracle Parsing Round-Trip Integrity

**User Story:** As a parity engineer, I want Oracle artifacts and network exports to survive a parse-then-serialize round trip, so that loading and exporting introduce no data corruption.

#### Acceptance Criteria

1. WHEN a valid Oracle artifact is provided, THE Oracle_Loader SHALL parse it into a comparison surface.
2. IF an Oracle artifact is malformed or missing a required field, THEN THE Oracle_Loader SHALL return a descriptive error identifying the artifact and the missing field.
3. THE Network_Stage SHALL serialize the network graph into the versioned `network.json` export.
4. WHEN a network graph is serialized to `network.json` and parsed back, THE SLAVV_Pipeline SHALL produce an equivalent network graph (round-trip property).

### Requirement 12: Certification Reporting

**User Story:** As a maintainer, I want the Parity_Harness to emit structured reports, so that I can distinguish certification verdicts from diagnostics.

#### Acceptance Criteria

1. WHEN the Parity_Harness completes a stage proof, THE Parity_Harness SHALL emit a Certification_Report recording missing count, extra count, and float agreement for the stage.
2. THE Certification_Report SHALL record ULP figures as diagnostics separate from the pass/fail verdict.
3. WHEN a maintainer requests strict floating-point comparison, THE Parity_Harness SHALL compare Continuous_Fields with bit-identical equality and report the result as a diagnostic.
4. THE Parity_Harness SHALL store disposable trial runs under `workspace/runs/` and promoted summaries under `workspace/reports/`.
