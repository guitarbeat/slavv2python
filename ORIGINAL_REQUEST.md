# Original User Request

## Initial Request — 2026-06-08T17:34:43-05:00

# Teamwork Project Prompt — Draft

> Status: Launched

Achieve 100% exact numerical parity between the Python SLAVV pipeline and the canonical MATLAB truth. The focus is on fixing downstream stages (Vertices, Edges, Network) sequentially, driven by failures in the `prove-exact` verification harness.

**Important Note**: The exact-route energy stage `resume-exact-run` is currently running in the background for `crop_M_exact`. You must wait for this run to finish or verify it has completed before starting your sequence proof, to avoid concurrent writers.

Working directory: `d:\2P_Data\Aaron\slavv2python`
Integrity mode: benchmark

## Requirements

### R1. Vertices Parity
Achieve strict zero missing/extra for the `vertices` stage on the crop harness (`180709_E_crop_M`). You must strictly follow the MATLAB 1:1 structure and use exact bitwise equality instead of `np.isclose`.

### R2. Edges Parity
Achieve strict zero missing/extra for the `edges` stage on the crop harness. Do not start this until Vertices parity is 100% resolved. No oracle-injection shims are allowed.

### R3. Network Parity
Achieve strict zero missing/extra for the `network` stage on the crop harness.

### R4. Canonical Volume Certification
Once the crop harness passes all stages, run the sequential proof gate on the full canonical volume (`180709_E_batch_190910-103039`) and achieve zero missing/extra across all four stages.

## Acceptance Criteria

### Exact Proof Sequence
- [ ] `python scripts/cli/parity_experiment.py prove-exact-sequence --source-run-root workspace/runs/oracle_180709_E/crop_M_exact --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact --oracle-root workspace/oracles/180709_E_crop_M` outputs 0 missing and 0 extra for all stages.
- [ ] `python scripts/cli/parity_experiment.py prove-exact-sequence --source-run-root workspace/runs/oracle_180709_E/phase1_cert_network --dest-run-root workspace/runs/oracle_180709_E/phase1_cert_network --oracle-root workspace/oracles/180709_E_batch_190910-103039` outputs 0 missing and 0 extra for all stages.
