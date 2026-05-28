# Parity Certification Guide

[Up: Reference Docs](../README.md) · [Parity Pre-Gate](PARITY_PRE_GATE.md)

This document provides step-by-step instructions for running a full mathematical parity certification between the SLAVV Python engine and a canonical MATLAB oracle.

For faster iteration before (or in parallel with) full-volume certification, see **[Parity Pre-Gate](PARITY_PRE_GATE.md)** (synthetic smoke → `180709_E_crop_M` → canonical `180709_E`).

---

## 🏗️ Certification Infrastructure

Certification is a developer-only workflow that uses the `scripts/cli/parity_experiment.py` harness to compare Python's output against preserved MATLAB truth vectors.

### Key Directories
- `workspace/oracles/`: Canonical MATLAB results (the "Ground Truth").
- `workspace/runs/`: Workspace for trial executions.
- `workspace/reports/`: Promoted, versioned certification reports.

---

## 🚀 Running a Certification Loop

### 1. Initialize a Parity Run
Create a structured run directory from a real dataset and a MATLAB oracle.
```powershell
python scripts/cli/parity_experiment.py init-exact-run `
  --dataset-root workspace/datasets/<dataset_id> `
  --oracle-root workspace/oracles/<oracle_id> `
  --dest-run-root workspace/runs/cert_trial_v1 `
  --stop-after network
```

### 2. Verify Preflight
Ensure the destination is correctly populated with oracle references and parameters.
```powershell
python scripts/cli/parity_experiment.py preflight-exact `
  --source-run-root workspace/runs/cert_trial_v1 `
  --oracle-root workspace/oracles/<oracle_id> `
  --dest-run-root workspace/runs/cert_trial_v1
```

### 3. Run the Exact Proof (sequential gates)

Phase 1 certification requires **strict zero** missing/extra per stage, in order:

`energy` → `vertices` → `edges` → `network`

Run each stage (or use `--stage all`, which compares all four in certification order):

```powershell
python scripts/cli/parity_experiment.py prove-exact `
  --source-run-root workspace/runs/cert_trial_v1 `
  --oracle-root workspace/oracles/<oracle_id> `
  --dest-run-root workspace/runs/cert_trial_v1 `
  --stage energy

# Repeat for vertices, edges, network — or use --stage all after all checkpoints exist
```

On Windows hosts where zarr rename fails, use `--energy-storage-format npy` on `init-exact-run` (see `docs/reference/backends/ZARR_ENERGY_STORAGE.md`).

---

## 📊 Interpreting the Results

The authoritative report is `workspace/runs/<run_id>/03_Analysis/exact_proof.json` (and `.txt`).

### Core Metrics
- **Stage passed**: Each compared field matches the oracle at strict equality (`passed: true` in `exact_proof.json`).
- **Missing / extra**: For edges, pair-level missing and extra counts must be **zero** for certification (not a percentage threshold).
- **Sequential gating**: If energy or vertices fail, do not claim downstream stages certified on that run.

Legacy `comparison_report.json` may still exist for older runs; prefer `exact_proof.json` for certification decisions.

---

## 🏆 Promoting to Official Certification

When all four stages pass with zero missing/extra, promote the run summary:

```powershell
# Copy the report to the authoritative index
Copy-Item workspace/runs/cert_trial_v1/report.json workspace/reports/CERTIFICATION_V0.1.0.json

# Update the Roadmap
# (Edit docs/ROADMAP.md to reflect the new High-Water Mark)
```

---

## 🛡️ Parity Guardrails

To maintain certification, every change to `global_watershed.py` or `common.py` must:
1.  **Maintain float64 precision**: No downcasting to `float32` in the critical path.
2.  **Use exact tie-breaking**: Compare energies with `==`, never `np.isclose`.
3.  **Preserve scanline order**: Ensure local strel indices follow the Fortran-order linear index priority.
