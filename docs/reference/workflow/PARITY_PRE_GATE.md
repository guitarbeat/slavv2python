# Parity Pre-Gate

[Up: Reference Docs](../README.md) · [Parity Certification Guide](PARITY_CERTIFICATION_GUIDE.md) · [ADR 0009](../../adr/0009-parity-pre-gate-tiers.md)

Operator workflow for the three-tier parity pre-gate agreed in ADR 0009. Terms are defined in [AGENTS.md](../../AGENTS.md) (**Parity Pre-Gate**, **Synthetic Fixture Volume**, **Crop Harness Volume**, **Canonical Volume**, **Certification**).

---

## Tier overview

| Tier | Volume | Oracle | `prove-exact` | Claim |
|------|--------|--------|---------------|--------|
| 1 — Synthetic | Python-generated TIFF | None | No | CI / harness smoke only |
| 2 — Crop harness | `180709_E_crop_M` (64×256×256 Z×Y×X) | `workspace/oracles/180709_E_crop_M` | Yes, strict zero, sequential | Harness confidence only |
| 3 — Canonical | Full `180709_E` | `180709_E_batch_190910-103039` | Yes, strict zero, sequential | Phase 1 exact-route **Certification** |

Tiers 2 and 3 use the **same** success bar (zero missing / zero extra per stage). Only tier 3 supports the Phase 1 certification milestone.

---

## Tier 1 — Synthetic fixture (CI smoke)

**Goal:** Fast regression that the pipeline runs on non-trivial topology without neurovasc-db or MATLAB.

- Extend `slavv_python/utils/synthetic.py` with at least one richer geometry (e.g. Y-junction or crossing tubes) in addition to the straight tube.
- **CI entry points:** `tests/integration/test_paper_profile_ci.py` (paper profile) and `tests/integration/parity/test_parity_pre_gate_tier1.py` (matlab_compat edges smoke + watershed LUT seam).
- Do **not** run `prove-exact` on synthetic data unless a dedicated MATLAB oracle is created for that volume (out of scope).

---

## Tier 2 — Crop harness (`180709_E_crop_M`)

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
python scripts/cli/export_180709_crop_m.py --write-metadata
```

Default output: `workspace/scratch/180709_E_crop_M/180709_E_crop_M.tif` plus `180709_E_crop_M.tif.roi.json`.

**Promoted dataset** (content-addressed hash dir):

- `workspace/datasets/0cdf88e930482e9eb818963da22846c43b53b531582bf3aed83678b549863d06/01_Input/180709_E_crop_M.tif`
- Use this path as `--dataset-root` for `init-exact-run` (the hash directory, not `180709_E_crop_M` as the folder name).

### Oracle (MATLAB truth)

Run vectorization with the released MATLAB tree under `external/Vectorization-Public/` (submodule). The repo ships a headless driver that imports canonical `180709_E` settings and writes a new `batch_*` under `workspace/scratch/matlab_crop_batches/`:

```powershell
matlab.exe -batch "run('scripts/matlab/vectorize_180709_E_crop_M.m')"
```

Requirements:

- Crop TIFF at the promoted dataset path (see **Promoted dataset** below).
- Canonical settings mats under `workspace/oracles/180709_E_batch_190910-103039/.../batch_190910-103039_canonical/settings/`.
- `VertexCuration` / `EdgeCuration` must be `'auto'` (not `'none'`).

When `vectors/` is populated under the new `batch_<timestamp>/`, promote:

```powershell
python scripts/cli/parity_experiment.py promote-oracle `
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
python scripts/cli/parity_experiment.py promote-dataset `
  --dataset-file <path-to-180709_E_crop_M.tif> `
  --experiment-root workspace

python scripts/cli/parity_experiment.py promote-oracle `
  --matlab-batch-dir <path-to-crop-batch> `
  --oracle-root workspace/oracles/180709_E_crop_M `
  --dataset-file <path-to-180709_E_crop_M.tif> `
  --oracle-id 180709_E_crop_M
```

Do **not** reuse `180709_E_batch_190910-103039` mats spatially cropped in Python — the oracle must come from MATLAB on the identical crop volume.

### Exact-route run and proof

```powershell
python scripts/cli/parity_experiment.py init-exact-run `
  --dataset-root workspace/datasets/<crop_dataset_id> `
  --oracle-root workspace/oracles/180709_E_crop_M `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --stop-after network `
  --energy-storage-format npy
```

Sequential certification on that run root:

```powershell
python scripts/cli/parity_experiment.py prove-exact-sequence `
  --source-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact `
  --oracle-root workspace/oracles/180709_E_crop_M
```

Reports: per-stage `03_Analysis/exact_proof_<stage>.json` and summary `03_Analysis/exact_proof.json`.

---

## Tier 3 — Canonical (Phase 1 milestone)

Unchanged from [PARITY_CERTIFICATION_GUIDE.md](PARITY_CERTIFICATION_GUIDE.md):

- Full `180709_E`, oracle `180709_E_batch_190910-103039`
- Native exact route, no vertex injection
- Phase 1 claim only after all four stages pass on **that** run

**Parallelism:** Crop harness work may run while a canonical run (e.g. `phase1_cert_network`) is in progress or resumed. Passing crop does not replace canonical certification.

**Preflight** before a long exact-route run (memory gate + params audit):

```powershell
python scripts/cli/parity_experiment.py preflight-exact `
  --source-run-root workspace/runs/oracle_180709_E/phase1_cert_network `
  --dest-run-root workspace/runs/oracle_180709_E/phase1_cert_network `
  --dataset-root workspace/datasets/<canonical_dataset_hash> `
  --oracle-root workspace/oracles/180709_E_batch_190910-103039
```

**Resume canonical run** when `init-exact-run` reports the seed run is still active but no Python process is running:

```powershell
python scripts/cli/parity_experiment.py resume-exact-run `
  --dest-run-root workspace/runs/oracle_180709_E/phase1_cert_network `
  --oracle-root workspace/oracles/180709_E_batch_190910-103039 `
  --stop-after network
```

Or `init-exact-run ... --resume` with the same `--dataset-root`, `--oracle-root`, and `--dest-run-root` as the original bootstrap.

Legacy helper `scripts/cli/resume_pipeline_run.py` delegates to `resume-exact-run` when `99_Metadata/experiment_provenance.json` exists.

---

## Related plans

- Phase 1 spec (requirements + plan): `docs/plans/phase-1-exact-route-spec.md`
