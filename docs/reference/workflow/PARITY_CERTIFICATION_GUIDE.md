# Parity Certification Guide

[Up: Reference Docs](../README.md) · [Parity Pre-Gate](PARITY_PRE_GATE.md)

This document provides step-by-step instructions for running a full mathematical parity certification between the SLAVV Python engine and a canonical MATLAB oracle.

For faster iteration before (or in parallel with) full-volume certification, see **[Parity Pre-Gate](PARITY_PRE_GATE.md)** (synthetic smoke → `180709_E_crop_M` → canonical `180709_E`).

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
Test-Path workspace/oracles/180709_E_batch_190910-103039/03_Analysis/normalized/oracle/energy.pkl
```

As of 2026-06-03, this artifact is present and readable:

- `energy`: `(64, 512, 512)` `float64`
- `scale_indices`: `(64, 512, 512)` `int64`
- `energy_4d`: empty placeholder
- `lumen_radius_microns`: `(99,)` `float64`
- payload sidecar SHA-256: `4696f05449541b6919d514b59705607eeb10258c67d5e466c52be83f73a43a9c`

If it is missing in a fresh workspace, materialize it from the canonical MATLAB batch. The current batch has an extensionless HDF5 energy volume, so this does not require rerunning MATLAB. For an existing Oracle root, use the Oracle Artifact Maintenance command to write only missing normalized artifacts and verify they are readable:

```powershell
slavv parity ensure-oracle-artifacts `
  --oracle-root workspace/oracles/180709_E_batch_190910-103039 `
  --stage energy
```

For a brand-new oracle root, use the full promotion command:

```powershell
slavv parity promote-oracle `
  --matlab-batch-dir workspace/oracles/180709_E_batch_190910-103039/01_Input/matlab_results/batch_190910-103039_canonical `
  --oracle-root workspace/oracles/180709_E_batch_190910-103039 `
  --dataset-file workspace/datasets/771eb62fd1322cf59e24f056aff2692b3375b94ce6dc9b25744428d4dbf1e353/01_Input/180709_E.tif `
  --oracle-id 180709_E_batch_190910-103039
```

The prepared canonical rerun command file is:

```powershell
workspace/scratch/phase1_cert_network_rerun_from_energy.ps1
```

### 2. Verify Preflight
Ensure the destination is correctly populated with oracle references and parameters.
```powershell
slavv parity preflight-exact `
  --source-run-root workspace/runs/cert_trial_v1 `
  --oracle-root workspace/oracles/<oracle_id> `
  --dest-run-root workspace/runs/cert_trial_v1
```

### 3. Run the Exact Proof (sequential gates)

Phase 1 certification requires **strict zero** missing/extra per stage, in order:

`energy` → `vertices` → `edges` → `network`

Run each stage, or use the sequential helper (stops on first failure and writes per-stage JSON under `03_Analysis/`):

```powershell
slavv parity prove-exact-sequence `
  --source-run-root workspace/runs/<cert_run> `
  --dest-run-root workspace/runs/<cert_run> `
  --oracle-root workspace/oracles/<oracle_id>
```

Alternatively run each stage manually (or use `--stage all`, which compares all four in certification order):

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
- **Missing / extra**: For edges, pair-level missing and extra counts must be **zero** for certification (not a percentage threshold).
- **Sequential gating**: If energy or vertices fail, do not claim downstream stages certified on that run.

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
