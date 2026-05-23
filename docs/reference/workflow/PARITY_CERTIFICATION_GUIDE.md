# Parity Certification Guide

[Up: Reference Docs](../README.md)

This document provides step-by-step instructions for running a full mathematical parity certification between the SLAVV Python engine and a canonical MATLAB oracle.

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

### 3. Run the Exact Proof
Execute the comparison and generate match rate metrics.
```powershell
python scripts/cli/parity_experiment.py prove-exact `
  --source-run-root workspace/runs/cert_trial_v1 `
  --oracle-root workspace/oracles/<oracle_id> `
  --dest-run-root workspace/runs/cert_trial_v1 `
  --stage all
```

---

## 📊 Interpreting the Results

The `prove-exact` command produces a summary in the console and a detailed JSON report at `workspace/runs/<run_id>/03_Analysis/comparison_report.json`.

### Core Metrics
- **Matched Pairs**: Number of edge connections bit-identical between Python and MATLAB.
- **Match Rate**: Target is **>95%** for certification.
- **Missing Pairs**: MATLAB connections not found in Python.
- **Extra Pairs**: Python connections not found in MATLAB (usually due to filtering differences).

---

## 🏆 Promoting to Official Certification

Once a match rate of >95% is achieved, promote the run to a versioned baseline.

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
