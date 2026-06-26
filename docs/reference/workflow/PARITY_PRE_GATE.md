# Parity Pre-Gate

[Up: Reference Docs](../README.md) · [Parity Certification Guide](PARITY_CERTIFICATION_GUIDE.md) · [ADR 0009](../../adr/0009-parity-pre-gate-tiers.md)

Operator workflow for the three-tier parity pre-gate agreed in ADR 0009. Terms are defined in [AGENTS.md](../../../AGENTS.md) (**Parity Pre-Gate**, **Synthetic Fixture Volume**, **Crop Harness Volume**, **Canonical Volume**, **Certification**).

---

## Tier overview

| Tier | Volume | Oracle | `prove-exact` | Claim |
|------|--------|--------|---------------|--------|
| 1 — Synthetic | Python-generated TIFF | None | No | CI / harness smoke only |
| 2 — Crop harness | `180709_E_crop_M` (64×256×256 Z×Y×X) | `workspace/oracles/180709_E_crop_M` | Yes, strict zero, sequential | Harness confidence only |
| 3 — Canonical | Full `180709_E` | `180709_E_batch_190910-103039` | Yes, strict zero, sequential | Phase 1 exact-route **Certification** |

Tiers 2 and 3 use the **same** success bar: for **energy/vertices**, strict zero (discrete/topological — zero missing / zero extra per stage) + `np.allclose` (continuous floats, [ADR 0011](../../adr/0011-energy-float-certification-policy.md)); for **edges/network**, the spatial bars in [ADR 0012](../../adr/0012-edge-watershed-parity-bar.md) (edges: voxel ownership-map + trace tolerance; network: endpoint-pair/bifurcation multisets + sub-voxel trace tolerance), since the watershed's exact pair-set is chaotically order-sensitive. Only tier 3 supports the Phase 1 certification milestone.

Check [EXACT_PROOF_FINDINGS.md](../core/EXACT_PROOF_FINDINGS.md) before Tier 3 proof. The canonical oracle must have a loadable energy artifact before `prove-exact-sequence` can certify the full volume.

---

## Tier 1 — Synthetic fixture (CI smoke)

**Goal:** Fast regression that the pipeline runs on non-trivial topology without neurovasc-db or MATLAB.

- Extend `slavv_python/utils/synthetic.py` with at least one richer geometry (e.g. Y-junction or crossing tubes) in addition to the straight tube.
- **CI entry points:** `tests/integration/test_paper_profile_ci.py` (paper profile) and `tests/integration/parity/test_parity_pre_gate_tier1.py` (matlab_compat edges smoke + watershed LUT seam).
- **Component parity (CI):** `tests/unit/pipeline/energy/test_matlab_linspace_table.py` (checked-in MATLAB mesh table), `tests/unit/parity/test_parity_harness.py`, `tests/unit/parity/test_random_component_parity.py`.
- **Random component differential (self-hosted):** [`tests/support/random_component_parity.py`](../../../tests/support/random_component_parity.py) — see [PARITY_RANDOM_COMPONENT_SUITE.md](PARITY_RANDOM_COMPONENT_SUITE.md) and [ADR 0010](../../adr/0010-random-component-parity-suite.md). Structural gate on linspace, `interp3`, and Energy shape/valid/coordinates; Hessian float ULP is advisory only.
- **Crop voxel probes (local):** `tests/unit/pipeline/energy/test_voxel_probe_regression.py` via [`tests/support/parity_harness.py`](../../../tests/support/parity_harness.py) — see [tests/README.md](../../../tests/README.md#matlabpython-bit-parity-testing).
- Do **not** run `prove-exact` on synthetic data unless a dedicated MATLAB oracle is created for that volume (out of scope).

---

## Tier 2 — Crop harness (`180709_E_crop_M`)

### Monitoring Long-Running Jobs

For long parity experiments (especially the energy stage), pass `--monitor` to `slavv parity launch-exact-run` (or `resume-exact-run`) to enable background job tracking and desktop notifications.

See [PARITY_JOB_MONITORING.md](PARITY_JOB_MONITORING.md) for the full command reference (starting monitored jobs, listing/inspecting job history, and daemon status).

---

### ROI definition (tier M)

Source volume: promoted dataset `180709_E.tif`, shape **64 × 512 × 512** (Z × Y × X).

| Axis | Full | Crop (centered) | Slice |
|------|------|-----------------|-------|
| Z | 64 | 64 | `[0:64)` |
| Y | 512 | 256 | `[128:384)` |
| X | 512 | 256 | `[128:384)` |

**Crop shape:** 64 × 256 × 256.

Export a TIFF with the repo script (writes ROI metadata when requested):

```powershell
slavv parity export-crop --write-metadata
```

Default output: `workspace/scratch/180709_E_crop_M/180709_E_crop_M.tif` plus `180709_E_crop_M.tif.roi.json`.

**Promoted dataset** (content-addressed hash dir):

- `workspace/datasets/0cdf88e930482e9eb818963da22846c43b53b531582bf3aed83678b549863d06/01_Input/180709_E_crop_M.tif`
- Use this path as `--dataset-root` for `init-exact-run` (the hash directory, not `180709_E_crop_M` as the folder name).

### Oracle (MATLAB truth)

Run vectorization with the released MATLAB tree under `external/Vectorization-Public/` (submodule). The repo ships a headless driver that imports canonical `180709_E` settings and writes a new `batch_*` under `workspace/scratch/matlab_crop_batches/`:

```powershell
matlab.exe -batch "run('workspace/scratch/matlab/vectorize_180709_E_crop_M.m')"
```

Requirements:

- Crop TIFF at the promoted dataset path (see **Promoted dataset** below).
- Canonical settings mats under `workspace/oracles/180709_E_batch_190910-103039/.../batch_190910-103039_canonical/settings/`.
- `VertexCuration` / `EdgeCuration` must be `'auto'` (not `'none'`).

When `vectors/` is populated under the new `batch_<timestamp>/`, promote:

```powershell
slavv parity promote-oracle `
  --matlab-batch-dir workspace/scratch/matlab_crop_batches/batch_<timestamp> `
  --oracle-root workspace/oracles/180709_E_crop_M `
  --dataset-file workspace/datasets/0cdf88e930482e9eb818963da22846c43b53b531582bf3aed83678b549863d06/01_Input/180709_E_crop_M.tif `
  --oracle-id 180709_E_crop_M
```

Do **not** reuse database vectorization zips or spatially crop the full `180709_E` oracle in Python — the oracle must come from MATLAB on the identical crop volume.

Legacy steps (manual batch layout):

1. Run MATLAB vectorization on the **crop TIFF only** (same parameter family as full `180709_E`, on that subvolume).
2. Produce a single timestamp-matched `batch_*` tree (one artifact per stage + `settings/`).
3. Promote:

```powershell
slavv parity promote-dataset `
  --dataset-file <path-to-180709_E_crop_M.tif> `
  --experiment-root workspace

slavv parity promote-oracle `
  --matlab-batch-dir <path-to-crop-batch> `
  --oracle-root workspace/oracles/180709_E_crop_M `
  --dataset-file <path-to-180709_E_crop_M.tif> `
  --oracle-id 180709_E_crop_M
```

Do **not** reuse `180709_E_batch_190910-103039` mats spatially cropped in Python — the oracle must come from MATLAB on the identical crop volume.

### Exact-route run and proof

```powershell
slavv parity init-exact-run `
  --dataset-root workspace/datasets/<crop_dataset_id> `
  --oracle-root workspace/oracles/180709_E_crop_M `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --stop-after network `
  --energy-storage-format npy
```

Sequential certification on that run root:

```powershell
slavv parity prove-exact-sequence `
  --source-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M
```

Reports: per-stage `03_Analysis/exact_proof_<stage>.json` and summary `03_Analysis/exact_proof.json`.

### Active crop closure workflow

Use this when a crop rerun is already active under `workspace/runs/oracle_180709_E/crop_M_exact`:

```powershell
# Watch the crop run from the consolidated operations console
slavv monitor --run-dir workspace/runs/oracle_180709_E/crop_M_exact

# Or print one non-interactive snapshot
slavv monitor --run-dir workspace/runs/oracle_180709_E/crop_M_exact --once

# When it exits, prove energy first
slavv parity prove-exact `
  --source-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M `
  --stage energy

# If energy passes after an energy-only rerun, refresh downstream checkpoints
slavv parity resume-exact-run `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M `
  --force-rerun-from vertices `
  --stop-after network `
  --skip-preflight

# Then run the sequential crop gate on fresh checkpoints
slavv parity prove-exact-sequence `
  --source-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M
```

If the energy proof still fails, inspect the first failing field before changing code. Current diagnostics are indexed in `workspace/scratch/parity_energy_diagnostics.md`.

---

## Tier 3 — Canonical (Phase 1 milestone)

Unchanged from [PARITY_CERTIFICATION_GUIDE.md](PARITY_CERTIFICATION_GUIDE.md):

- Full `180709_E`, oracle `180709_E_batch_190910-103039`
- Native exact route, no vertex injection
- Phase 1 claim only after all four stages pass on **that** run

**Parallelism:** Crop harness work may run while a canonical run (e.g. `phase1_cert_network`) is in progress or resumed. Passing crop does not replace canonical certification.

**Preflight** before a long exact-route run (memory gate + params audit):

```powershell
slavv parity preflight-exact `
  --source-run-root workspace/runs/oracle_180709_E/phase1_cert_network `
  --dest-run-root workspace/runs/oracle_180709_E/phase1_cert_network `
  --dataset-root workspace/datasets/<canonical_dataset_hash> `
  --oracle-root workspace/oracles/180709_E_batch_190910-103039
```

**Resume canonical run** when `init-exact-run` reports the seed run is still active but no Python process is running:

```powershell
slavv parity resume-exact-run `
  --dest-run-root workspace/runs/oracle_180709_E/phase1_cert_network `
  --oracle-root workspace/oracles/180709_E_batch_190910-103039 `
  --stop-after network
```

Or `init-exact-run ... --resume` with the same `--dataset-root`, `--oracle-root`, and `--dest-run-root` as the original bootstrap.

Use `slavv parity resume-exact-run` to resume an interrupted run when `99_Metadata/experiment_provenance.json` exists.

---

## Related plans

- Phase 1 spec (requirements + plan): `docs/plans/phase-1-exact-route-spec.md`
