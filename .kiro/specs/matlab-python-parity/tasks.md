# Implementation Plan: MATLAB-Python Parity

## Overview

All four stages are certified on the crop harness (`180709_E_crop_M_v2`). The remaining
work is:

1. Writing the 17 property-based tests from the design's Correctness Properties section
2. Auditing and completing any Exact Route implementation gaps (float64 enforcement, grid
   alignment, oracle loader conventions)
3. Running and passing the three-tier pre-gate sequence through canonical certification
4. Producing the final `CertificationReport` for Phase 1

The Python pipeline code largely exists. Tasks focus on gaps, tests, and the certification
run sequence. All test files go under `tests/unit/` or `tests/integration/` per
`tests/README.md`.

---

## Tasks

- [x] 1. Audit and close Exact Route implementation gaps
  - [x] 1.1 Audit float64 enforcement across all four stage computation paths
    - Read `pipeline/energy/matlab_get_energy_v202_chunked.py`,
      `pipeline/vertices/detection.py`, `pipeline/edges/matlab_get_edges_by_watershed.py`,
      `pipeline/network/manager.py`
    - Confirm all intermediate arrays (energies, coordinates, radii, distance penalties,
      suppression factors, strand geometry) are `float64` before any persistence coercion
    - Add explicit `astype(np.float64)` or dtype enforcement at any conversion site that
      is missing it; do NOT change persistence dtype (energy persisted as float32 is
      intentional per design)
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 1.2 Verify Fortran-order grid and orientation transitions in watershed
    - Read `pipeline/edges/matlab_get_edges_by_watershed.py` and
      `pipeline/edges/matlab_watershed_heap.py`
    - Confirm `vertex_index_map` and `energy_map` enter the watershed as
      `np.asfortranarray` with shape `[Y, X, Z]`
    - Confirm the `[Z,Y,X] → [Y,X,Z]` transpose is applied exactly once on entry and
      reversed exactly once on artifact persistence (double-transpose fix `e9dcc141` is
      in place — verify no regression)
    - _Requirements: 3.1, 3.3_

  - [x] 1.3 Verify round-half-away-from-zero rounding in vertex painting and candidate
          filtering
    - Read `pipeline/vertices/painting.py` and the candidate scan in `detection.py`
    - Confirm all `.5`-boundary coordinate rounding uses `floor(x + 0.5)` and not
      Python's built-in `round()` (banker's rounding)
    - Add a `slavv_round` helper in `slavv_python/utils/math_utils.py` if a shared
      implementation does not already exist; replace inline rounding calls
    - _Requirements: 3.4_

  - [x] 1.4 Verify oracle loader index-shift and axis-reversal conventions
    - Read `analytics/parity/oracle/oracle_artifacts.py`
    - Confirm exactly one `index - 1` shift is applied to MATLAB scale-index artifacts
      (no double-shift)
    - Confirm exactly one `np.flip` / axis-reversal is applied to v7.3 HDF5 artifacts to
      recover `[Z,Y,X]` physical order
    - Confirm curated vertex energy recovery uses `vertices.mat` raw energies (not the
      rank-ramp from `curated_vertices.mat`)
    - _Requirements: 10.3, 10.4_

  - [x] 1.5 Verify parameter whitelist includes `comparison_exact_network`
    - Read `slavv_python/utils/validation.py` (or wherever `RunContext.prepare()` strips
      params)
    - Confirm `comparison_exact_network` is in the allowed-keys whitelist; add it if
      absent
    - Confirm conflict painting is disabled when `comparison_exact_network=True` in
      `pipeline/edges/selection.py`
    - _Requirements: 1.2, 6.5_

- [x] 2. Checkpoint — confirm implementation gaps are closed
  - Run `python -m pytest tests/unit/pipeline/ tests/unit/parity/ -m "unit" -x` and confirm
    all existing tests pass before starting PBT work.
  - Ask the user if questions arise.

- [x] 3. Write property-based tests: pipeline computation invariants (Properties 1–5)
  - [x] 3.1 Write `tests/unit/pipeline/test_float64_dtype_invariant.py` — Property 1
    - Use `hypothesis` with `@given(st.integers(2,16), st.integers(2,16), st.integers(2,8))`
      to generate synthetic volume shapes
    - Use `unittest.mock` to intercept intermediate arrays after energy, vertex, edge, and
      network computation steps; assert `arr.dtype == np.float64` for all continuous arrays
    - Minimum `@settings(max_examples=100)`
    - Include comment tag: `# Feature: matlab-python-parity, Property 1: Float64 Computation Invariant`
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 3.2 Write `tests/unit/pipeline/test_fortran_order_grid.py` — Property 2
    - Use `hypothesis` with `@given(st.integers(2,32), st.integers(2,32), st.integers(2,16))`
      for `[Y,X,Z]` shapes; assert `np.isfortran(np.asfortranarray(arr)) == True` and
      shape is `(Y,X,Z)` after the watershed entry transpose
    - Include comment tag: `# Feature: matlab-python-parity, Property 2: Fortran-Order Grid Invariant`
    - _Requirements: 3.1_

  - [x] 3.3 Write `tests/unit/pipeline/test_fortran_tie_breaking.py` — Property 3
    - Use `hypothesis` to generate arbitrary 3D `[Y,X,Z]` grids with at least two
      voxels sharing the minimum energy value; assert the winner is the voxel with the
      lower `np.ravel_multi_index(..., order='F')` index
    - Include comment tag: `# Feature: matlab-python-parity, Property 3: Fortran-Order Tie-Breaking`
    - _Requirements: 3.2_

  - [x] 3.4 Write `tests/unit/pipeline/test_orientation_round_trip.py` — Property 4
    - Use `hypothesis` `st.arrays` with random shapes and float64 values; apply the
      `[Y,X,Z]→[Z,Y,X]→[Y,X,Z]` transpose sequence; assert the result equals the
      original with `np.array_equal`
    - Include comment tag: `# Feature: matlab-python-parity, Property 4: Orientation Persistence Round-Trip`
    - _Requirements: 3.3_

  - [x] 3.5 Write `tests/unit/pipeline/test_round_half_away_from_zero.py` — Property 5
    - Use `hypothesis` `st.floats` filtered to values where `x - math.floor(x) == 0.5`
      (positive, negative, large magnitude); call `slavv_round(x)` and assert the result
      equals `math.floor(x) + 1`
    - Include comment tag: `# Feature: matlab-python-parity, Property 5: Round-Half-Away-from-Zero`
    - _Requirements: 3.4_

- [x] 4. Write property-based tests: energy engine and structuring element (Properties 6–7)
  - [x] 4.1 Write `tests/unit/pipeline/energy/test_matlab_linspace_property.py` — Property 6
    - Extend or complement the existing `test_matlab_linspace_table.py` with a
      hypothesis-driven property test using `@given(st.integers(2,200), st.floats(...))`
      varying N and endpoint ranges; assert `abs(python_val - matlab_ref_val) < 1e-14` at
      every grid point including coarse-cell boundaries
    - Name the new file distinctly from the existing table test to avoid confusion
    - Include comment tag: `# Feature: matlab-python-parity, Property 6: MATLAB Linspace Mesh Correctness`
    - _Requirements: 4.4_

  - [x] 4.2 Write `tests/unit/pipeline/test_structuring_element_membership.py` — Property 7
    - Use `hypothesis` `st.floats(min_value=0.5, max_value=20.0)` for radius `r`; call
      `ellipsoid_offsets(r, ...)` and assert every returned offset satisfies
      `sqrt(dy²+dx²+dz²) <= r` (float comparison) and no excluded offset satisfies it
    - Include comment tag: `# Feature: matlab-python-parity, Property 7: Structuring Element Float-Radius Membership`
    - _Requirements: 5.4_

- [x] 5. Write property-based tests: parity harness sequencing (Properties 8–11)
  - [x] 5.1 Write `tests/unit/parity/test_prove_exact_sequence_order.py` — Property 8
    - Mock the four stage evaluators in `ExactProofCoordinator`; use `hypothesis`
      to vary which stages pass/fail; assert that the mock call order is always
      `[energy, vertices, edges, network]` with no stage called before its predecessor
      has returned
    - Include comment tag: `# Feature: matlab-python-parity, Property 8: Sequential Stage Evaluation Order`
    - _Requirements: 8.1_

  - [x] 5.2 Write `tests/unit/parity/test_downstream_blocking.py` — Property 9
    - Use `hypothesis` `st.sampled_from(["energy","vertices","edges"])` to pick a
      failing stage X; mock the coordinator so stage X returns FAIL; assert all stages
      after X receive `verdict = "BLOCKED"` without their comparators being called
    - Include comment tag: `# Feature: matlab-python-parity, Property 9: Downstream Blocking on Stage Failure`
    - _Requirements: 8.2_

  - [x] 5.3 Write `tests/unit/parity/test_first_failing_field.py` — Property 10
    - Craft `CertificationReport` comparator results where one field fails; use
      `hypothesis` to vary which field in `[scale_indices, energy, lumen_radius_microns,
      positions, scales, energies, ownership_map, endpoint_pairs, bifurcations]` fails
      first; assert `report.first_failing_field` matches the injected failing field name
    - Include comment tag: `# Feature: matlab-python-parity, Property 10: First-Failing-Field Identification`
    - _Requirements: 8.4_

  - [x] 5.4 Write `tests/unit/parity/test_all_pass_certification.py` — Property 11
    - Mock all four stage comparators to return PASS; use `hypothesis` to vary
      mock float-agreement and missing/extra count values (all zeros); assert the
      aggregate verdict is `CERTIFIED`
    - Include comment tag: `# Feature: matlab-python-parity, Property 11: All-Pass Certification Verdict`
    - _Requirements: 9.2_

- [x] 6. Write property-based tests: oracle loader conventions (Properties 12–14, 16)
  - [x] 6.1 Write `tests/unit/parity/test_oracle_loader_index_shift.py` — Property 12
    - Use `hypothesis` `st.arrays(dtype=np.int32, shape=st.integers(1,100))` with values
      in `[1, 255]` range (MATLAB 1-based); call the oracle loader's index-shift path;
      assert every returned value equals `raw - 1` exactly
    - Include comment tag: `# Feature: matlab-python-parity, Property 12: Oracle Exactly-One Index Shift`
    - _Requirements: 10.3_

  - [x] 6.2 Write `tests/unit/parity/test_oracle_loader_axis_reversal.py` — Property 13
    - Use `hypothesis` `st.integers(1,8)` for each of three axis dimensions; construct a
      synthetic array; apply the loader's axis-reversal; apply it again; assert the
      double-reversal recovers the original (invertibility check)
    - Include comment tag: `# Feature: matlab-python-parity, Property 13: Oracle HDF5 Axis Reversal Round-Trip`
    - _Requirements: 10.4_

  - [x] 6.3 Write `tests/unit/parity/test_oracle_loader_error_messages.py` — Property 14
    - Use `hypothesis` `st.text()` for artifact path and field name; call the oracle
      loader with a malformed/missing-field artifact fixture; assert the raised exception
      message contains both the artifact path and the field name as substrings
    - Include comment tag: `# Feature: matlab-python-parity, Property 14: Oracle Loader Error Identification`
    - _Requirements: 11.2_

  - [x] 6.4 Write `tests/unit/parity/test_oracle_artifact_completeness.py` — Property 16
    - Build a minimal fixture oracle directory with stub artifacts for all four stages
      using `tmp_path`; call the oracle loader for each stage; assert all four surfaces
      are non-None and no `OracleMissingArtifactError` is raised
    - Include comment tag: `# Feature: matlab-python-parity, Property 16: Oracle Artifact Completeness`
    - _Requirements: 10.2_

- [x] 7. Write property-based tests: serialization and reporting (Properties 15, 17)
  - [x] 7.1 Write `tests/unit/schema/test_network_serialization_roundtrip.py` — Property 15
    - Use `hypothesis` to generate random `NetworkResult` instances (varying strand
      counts 0–20, bifurcation counts 0–10, vertex-degree arrays); serialize to a temp
      `network.json` via `NetworkExporter`; deserialize back; assert strand-endpoint-pair
      multiset, bifurcation multiset, and vertex-degree array are equal to the original
    - Include comment tag: `# Feature: matlab-python-parity, Property 15: Network Graph Serialization Round-Trip`
    - _Requirements: 11.4_

  - [x] 7.2 Write `tests/unit/parity/test_certification_report_fields.py` — Property 17
    - Use `hypothesis` to generate random pass/fail outcomes with varying `max_delta`,
      `pass_rate`, `missing_count`, `extra_count` values; construct `CertificationReport`
      objects; assert presence of `missing_count`, `extra_count`, `float_agreement`, and
      `diagnostics.ulp_figures`; assert `verdict` is determined only by parity bars and
      is not affected by ULP magnitude
    - Include comment tag: `# Feature: matlab-python-parity, Property 17: Certification Report Required Fields`
    - _Requirements: 12.1, 12.2_

- [x] 8. Checkpoint — Tier 1 CI gate
  - Run the full Tier 1 regression gate locally:
    ```
    python -m pytest tests/unit/pipeline/ tests/unit/parity/ tests/unit/schema/ -m "unit" -x
    python -m ruff format --check slavv_python tests
    python -m ruff check slavv_python tests
    python -m mypy
    ```
  - All 17 property test files must pass. Fix any failures before proceeding.
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Run Tier 2 crop harness and confirm all four stages certify
  - Run per-stage `prove-exact --stage <s>` for each stage against `workspace/oracles/180709_E_crop_M_v2`:
    ```
    slavv parity prove-exact --source-run-root workspace/runs/oracle_180709_E/crop_M_exact --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact --oracle-root workspace/oracles/180709_E_crop_M_v2 --stage energy
    slavv parity prove-exact --source-run-root workspace/runs/oracle_180709_E/crop_M_exact --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact --oracle-root workspace/oracles/180709_E_crop_M_v2 --stage vertices
    slavv parity prove-exact --source-run-root workspace/runs/oracle_180709_E/crop_M_exact --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact --oracle-root workspace/oracles/180709_E_crop_M_v2 --stage edges
    slavv parity prove-exact --source-run-root workspace/runs/oracle_180709_E/crop_M_exact --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact --oracle-root workspace/oracles/180709_E_crop_M_v2 --stage network
    ```
  - **Note**: Do NOT use `prove-exact-sequence` — it uses a strict field comparator that fails on edges due to accepted watershed order-sensitivity (ADR 0012). Use individual `prove-exact --stage <s>` which applies the correct ADR 0012 spatial bars for edges/network.
  - Confirm all four stages pass their bars (energy: 0 scale_indices mismatches +
    allclose; vertices: 0 positions/scales mismatches + allclose energies; edges:
    ownership-map ≥ 60% + trace tolerance; network: endpoint-pair + bifurcation
    multisets exact + trace tolerance)
  - Record the per-stage `exact_proof_<stage>.json` verdicts as Tier 2 evidence
  - _Requirements: 4.1, 4.2, 5.1, 5.2, 6.1, 6.2, 7.1, 7.2_

- [ ] 10. Launch canonical full-volume run (`180709_E`)
  - [ ] 10.1 Preflight check for canonical run
    - Run `slavv parity preflight-exact` against `workspace/oracles/180709_E_full_v2`
      and `workspace/runs/oracle_180709_E/canonical_full_v4` to confirm oracle artifacts
      are present for all four stages and run-dir is accessible
    - Confirm `comparison_exact_network=True` is in the run params and not stripped by
      the whitelist
    - _Requirements: 1.2, 9.1, 10.1, 10.2_

  - [ ] 10.2 Launch or resume the canonical full-volume exact-route run
    - Use `slavv parity resume-exact-run` (or `launch-exact-run`) with
      `--run-dir workspace/runs/oracle_180709_E/canonical_full_v4` and
      `--n-jobs <N>` for threaded energy parallelism (provably bit-exact; see
      `docs/solutions/parity/exact-energy-chunk-parallelism.md`)
    - Monitor with `slavv monitor --run-dir workspace/runs/oracle_180709_E/canonical_full_v4`
      and track energy chunk throughput via the joblib `Done N tasks` log
    - Confirm `max_voxels_per_node_energy` matches the MATLAB oracle batch lattice size
      (typically 6000) — chunk-boundary numerics diverge if this differs
    - _Requirements: 1.1, 1.3, 9.1_

- [ ] 11. Run `prove-exact-sequence` on canonical volume and certify Phase 1
  - [ ] 11.1 Run canonical `prove-exact-sequence`
    - Once the canonical run completes, execute:
      ```
      slavv parity prove-exact-sequence `
        --source-run-root workspace/runs/oracle_180709_E/canonical_full_v4 `
        --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v4 `
        --oracle-root workspace/oracles/180709_E_full_v2
      ```
    - All four stages must pass the same bars as Tier 2
    - _Requirements: 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 6.1, 6.2, 7.1, 7.2, 8.1, 8.2,
      9.2, 9.3_

  - [ ] 11.2 Verify `CertificationReport` structure and promote to `workspace/reports/`
    - Confirm `exact_proof.json` contains the aggregate `CERTIFIED` verdict and that
      per-stage `exact_proof_<stage>.json` files each contain `missing_count`,
      `extra_count`, `float_agreement`, `discrete_agreement`, `diagnostics.ulp_figures`,
      and `first_failing_field: null`
    - Copy the proof JSON files to `workspace/reports/canonical_180709_E_phase1/` as the
      promoted Phase 1 certification evidence
    - Update `docs/reference/core/EXACT_PROOF_FINDINGS.md` with the canonical run
      verdict, run root, oracle root, and date
    - _Requirements: 9.2, 12.1, 12.2, 12.3, 12.4_

- [ ] 12. Final checkpoint — Phase 1 complete
  - Re-run the Tier 1 regression gate to confirm no test regressions from the
    implementation gap fixes:
    ```
    python -m pytest tests/unit/ tests/integration/ -m "unit or integration" -x
    python -m ruff format --check slavv_python tests
    python -m ruff check slavv_python tests
    python -m mypy
    ```
  - Confirm `workspace/reports/canonical_180709_E_phase1/exact_proof.json` exists with
    `"verdict": "CERTIFIED"`
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP. No tasks in this
  plan are marked optional — all 17 PBTs are required for the spec's Correctness
  Properties coverage.
- Tasks 1.x are audits/fixes; if the audit confirms the implementation is already
  correct, the sub-task is satisfied with a brief comment in the commit message.
- Tier 2 (task 9) and Tier 3 (tasks 10–11) require `workspace/` oracle artifacts and
  a local machine — they are not CI tasks.
- The canonical energy run is multi-hour. Use `--n-jobs` for threaded parallelism
  (bit-exact; ordered merge). Never compute ETA from `resume_state.json` progress — use
  the joblib `Done N tasks` log as the leading indicator.
- Do not use raw edge-pair overlap as a parity metric. Use the ownership-map bar (edges)
  and endpoint-pair + bifurcation multisets (network) per ADR 0012.
- Each property test must include a comment tag matching the design document property
  number and title.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "1.4", "1.5"] },
    { "id": 1, "tasks": ["3.1", "3.2", "3.3", "3.4", "3.5", "4.1", "4.2"] },
    { "id": 2, "tasks": ["5.1", "5.2", "5.3", "5.4", "6.1", "6.2", "6.3", "6.4", "7.1", "7.2"] },
    { "id": 3, "tasks": ["10.1"] },
    { "id": 4, "tasks": ["10.2"] },
    { "id": 5, "tasks": ["11.1"] },
    { "id": 6, "tasks": ["11.2"] }
  ]
}
```
