# Parity Certification Guide

[Up: Reference Docs](../README.md) · [Parity Pre-Gate](PARITY_PRE_GATE.md)

This document provides step-by-step instructions for running a full mathematical parity certification between the SLAVV Python engine and a canonical MATLAB oracle.

For faster iteration before (or in parallel with) full-volume certification, see **[Parity Pre-Gate](PARITY_PRE_GATE.md)** (synthetic smoke → `180709_E_crop_M_v2` → canonical `180709_E`).

---

## 🏗️ Certification Infrastructure

Certification is a developer-only workflow that uses the `slavv parity` harness to compare Python's output against preserved MATLAB truth vectors.

Use `slavv monitor --run-dir <run_root>` as the primary operations console while
long parity runs are active. Deprecated watcher scripts under `scripts/` have
been removed so stale PID vs stale snapshot interpretation has one source of
truth.

### Key Directories
- `workspace/datasets/`: immutable input volumes plus manifests.
- `workspace/oracles/`: preserved MATLAB truth packages only.
- `workspace/runs/`: disposable Python reruns and developer diagnostics.
- `workspace/reports/`: promoted, versioned certification summaries.

### Run Layout

Each exact run uses the structured harness layout:

```text
runs/<run_id>/
  00_Refs/
  01_Params/
  02_Output/
  03_Analysis/
  99_Metadata/
```

- `01_Params/` owns `shared_params.json`, `python_derived_params.json`, and `param_diff.json`.
- `02_Output/` owns Python checkpoints and stage artifacts.
- `03_Analysis/` owns `exact_proof*.json`, text reports, normalized payloads, and hashes.
- `99_Metadata/` owns snapshots, manifests, and command provenance.

Treat `runs/` as disposable. Promote only durable proof summaries into
`workspace/reports/`. Treat `oracles/` as preserved MATLAB truth, not scratch
space.

---

## 🚀 Running a Certification Loop

### 1. Initialize a Parity Run
Create a structured run directory from a real dataset and a MATLAB oracle.
```powershell
slavv parity init-exact-run `
  --dataset-root workspace/datasets/<dataset_id> `
  --oracle-root workspace/oracles/<oracle_id> `
  --dest-run-root workspace/runs/cert_trial_v1 `
  --stop-after network
```

For long-running jobs, add the `--monitor` flag when you start the run with `slavv parity launch-exact-run` (or `resume-exact-run`) to enable automatic tracking and desktop notifications. See [PARITY_JOB_MONITORING.md](PARITY_JOB_MONITORING.md) for details.

### 1a. Ensure the canonical oracle has energy

Before proving full `180709_E`, confirm the canonical oracle has a normalized energy artifact:

```powershell
Test-Path workspace/oracles/180709_E_full_v2/03_Analysis/normalized/oracle/energy.pkl
```

The `full_v2` energy artifact is present per EXACT_PROOF_FINDINGS.md (energy `(64, 512, 512)` `float64`; canonical energy certified with 0 scale mismatches). Run `ensure-oracle-artifacts --stage energy` (below) to verify readability and capture the current payload hash in your run's `03_Analysis/` before proving. (The legacy `180709_E_batch_190910-103039` energy.pkl SHA-256 `4696f05449541b6919d514b59705607eeb10258c67d5e466c52be83f73a43a9c` is retained here only for historical reference.)

If it is missing in a fresh workspace, materialize it from the canonical MATLAB batch. The current batch has an extensionless HDF5 energy volume, so this does not require rerunning MATLAB. For an existing Oracle root, use the Oracle Artifact Maintenance command to write only missing normalized artifacts and verify they are readable:

```powershell
slavv parity ensure-oracle-artifacts `
  --oracle-root workspace/oracles/180709_E_full_v2 `
  --stage energy
```

For a brand-new oracle root, use the full promotion command:

```powershell
slavv parity promote-oracle `
  --matlab-batch-dir workspace/oracles/180709_E_full_v2/01_Input/matlab_results/batch_260626-125646 `
  --oracle-root workspace/oracles/180709_E_full_v2 `
  --dataset-file workspace/datasets/771eb62fd1322cf59e24f056aff2692b3375b94ce6dc9b25744428d4dbf1e353/01_Input/180709_E.tif `
  --oracle-id 180709_E_full_v2
```

The prepared canonical closure run is documented in **[.claude/HANDOFF.md](../../.claude/HANDOFF.md)** (`canonical_full_v5`, preflight from `canonical_full_v4`).

### 2. Verify Preflight
Ensure the destination is correctly populated with oracle references and parameters.
```powershell
slavv parity preflight-exact `
  --source-run-root workspace/runs/cert_trial_v1 `
  --oracle-root workspace/oracles/<oracle_id> `
  --dest-run-root workspace/runs/cert_trial_v1
```

### 3. Run the Exact Proof

Phase 1 closure ([ADR 0012 addendum](../adr/0012-edge-watershed-parity-bar.md#addendum-2026-07-06-phase-1-closure-bar-vs-strict-field-stretch)) uses **per-stage** gates in order:

`energy` → `vertices` → `edges` → `network`

| Stage | Bar |
|-------|-----|
| Energy, Vertices | Strict discrete + ADR 0011 `np.allclose` on floats |
| Edges, Network | ADR 0012 spatial bars (ownership-map / strand multisets + trace tolerance) |

**Recommended for Phase 1 closure** — run edges and network individually after fresh checkpoints exist on `canonical_full_v5`:

```powershell
slavv parity prove-exact `
  --source-run-root workspace/runs/oracle_180709_E/canonical_full_v5 `
  --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v5 `
  --oracle-root workspace/oracles/180709_E_full_v2 `
  --stage edges

slavv parity prove-exact `
  --source-run-root workspace/runs/oracle_180709_E/canonical_full_v5 `
  --dest-run-root workspace/runs/oracle_180709_E/canonical_full_v5 `
  --oracle-root workspace/oracles/180709_E_full_v2 `
  --stage network
```

`prove-exact-sequence` uses the **strict-field** comparator and may fail Edges/Network on watershed order-sensitivity even when ADR 0012 per-stage proofs pass. Use it for stretch milestone checks on crop, not as the Phase 1 ship gate.

```powershell
slavv parity prove-exact-sequence `
  --source-run-root workspace/runs/<cert_run> `
  --dest-run-root workspace/runs/<cert_run> `
  --oracle-root workspace/oracles/<oracle_id>
```

Alternatively run each stage manually (energy and vertices if not already certified):

```powershell
slavv parity prove-exact `
  --source-run-root workspace/runs/cert_trial_v1 `
  --oracle-root workspace/oracles/<oracle_id> `
  --dest-run-root workspace/runs/cert_trial_v1 `
  --stage energy

# Repeat for vertices, edges, network — or use --stage all after all checkpoints exist
```

On Windows hosts where zarr rename fails, use `--energy-storage-format npy` on `init-exact-run` (see `docs/reference/backends/ZARR_ENERGY_STORAGE.md`).

### 4. Resume an interrupted run

When `init-exact-run` seeded a run but the pipeline process stopped, resume with a **single** writer on the destination root:

```powershell
slavv parity resume-exact-run `
  --dest-run-root workspace/runs/<cert_run> `
  --oracle-root workspace/oracles/<oracle_id> `
  --stop-after network `
  --skip-preflight `
  --monitor
```

Or re-run `init-exact-run` with `--resume` and the same `--dataset-root`, `--oracle-root`, and `--dest-run-root`. Do not run `init-exact-run` and `resume-exact-run` concurrently on the same destination.

**Monitoring long runs:**

For parity experiments that take 4+ hours, use `--monitor` to enable automatic tracking:
- Background daemon tracks job progress
- Desktop notifications on completion/failure
- Prevents duplicate writers on same run directory
- Job history persists across terminal restarts

```powershell
# Check active monitored jobs
slavv jobs list

# View job history
slavv jobs history --run-dir workspace/runs/<cert_run>

# Interactive monitoring (real-time progress)
slavv monitor --run-dir workspace/runs/<cert_run>
```

See [PARITY_JOB_MONITORING.md](PARITY_JOB_MONITORING.md) for comprehensive monitoring documentation.

---

## 📊 Interpreting the Results

The authoritative report is `workspace/runs/<run_id>/03_Analysis/exact_proof.json` (and `.txt`).

### Core Metrics
- **Stage passed**: Each compared field matches the oracle — **strict equality** for discrete/topological fields, **`np.allclose(rtol=1e-7, atol=1e-9)`** for continuous float fields (energy, radii) per [ADR 0011](../../adr/0011-energy-float-certification-policy.md) — yielding `passed: true` in `exact_proof.json`. Use `--strict-floats` to force bit-identical float comparison (regression only).
- **Missing / extra**: Energy and vertices certify on strict **zero** missing/extra (discrete fields). **Edges and network do NOT use exact pair-set equality** — per [ADR 0012](../../adr/0012-edge-watershed-parity-bar.md) they certify on spatial bars: edges on **voxel ownership-map agreement** (Python `vertex_index_map` vs MATLAB) + per-edge trace tolerance; network on **strand endpoint-pair + bifurcation multisets** (exact, order-independent) + per-strand geometry within a sub-voxel trace tolerance. Raw edge-pair overlap is misleading (it was inflated by the now-fixed double-transpose bug) and is not a certification metric.
- **Sequential gating**: If energy or vertices fail, do not claim downstream stages certified on that run.

### Reading a result honestly (don't fool yourself)

A green proof is necessary, not sufficient. Before trusting one, run these checks —
each maps to a real near-miss on this project:

- **Ask what the test case cannot distinguish.** The crop (`180709_E_crop_M`) has
  **Y = X = 256**, so an axis swap that mis-orients the volume to `[Y, Z, X]` is
  *invisible* on it — the crop certified while the full volume was silently wrong
  (see [solutions/parity/resume-energy-orientation](../../solutions/parity/resume-energy-orientation.md)).
  **A pass on a symmetric/small/single-case fixture is weak evidence for the
  asymmetric full volume.** Certify on a case whose Z, Y, X are all distinct.
- **Confirm the shape/orientation before the values.** Check the Python checkpoint
  shape equals the oracle energy size (`(64,512,512)` for full `180709_E`). A
  `ValueError: Energy and scale arrays must share shape` (or a transposed shape)
  means orientation, not a value mismatch — fix that first.
- **A metric can be right for the wrong reason.** Edge **pair-overlap** looked high
  while traces ran on a scrambled grid; the mechanistically-grounded
  **ownership-map %** is the real bar (ADR 0012). Prefer metrics tied to the thing
  you care about.
- **Decompose a FAIL; don't read the headline.** "Energy ULP fail (37k voxels)" was
  actually max \|Δ\| = 1.99×10⁻¹¹ — a near-zero artifact of pure-ULP, which is why
  the gate is `np.allclose` (ADR 0011). Report the field, the worst voxel, and the
  absolute delta.
- **Hypothesis vs verified.** Do not record a root cause as confirmed until a cheap
  *disconfirming* test survives (the findings log has several "CONFIRMED → RETRACTED"
  entries that a refuting check would have prevented).
- **When deduction contradicts the evidence twice, get ground truth** — print the
  actual arrays/shapes or simulate the real code path; don't reason in circles.

Legacy `comparison_report.json` may still exist for older runs; prefer `exact_proof.json` for certification decisions.

---

## 🏆 Promoting to Official Certification

When all four stages pass with zero missing/extra, promote the run summary:

```powershell
# Copy the exact proof summary to the authoritative index
Copy-Item workspace/runs/cert_trial_v1/03_Analysis/exact_proof.json workspace/reports/CERTIFICATION_V0.1.0.json

# Update the live status log
# (Edit docs/reference/core/EXACT_PROOF_FINDINGS.md to record the milestone)
```

---

## 🛡️ Parity Guardrails

To maintain certification, every change to `matlab_get_edges_by_watershed.py` or `matlab_indexing.py` must:
1.  **Maintain float64 precision**: No downcasting to `float32` in the critical path.
2.  **Use exact tie-breaking**: Compare energies with `==`, never `np.isclose`.
3.  **Preserve scanline order**: Ensure local strel indices follow the Fortran-order linear index priority.
