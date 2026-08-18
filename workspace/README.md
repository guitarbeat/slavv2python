# Workspace

See [Experiment Root](../AGENTS.md#experiment-root).

**Purpose:** On-disk Oracles, datasets, live Claim Run Root / crop guard /
stretch dest, and archived proofs. GitHub does not carry those binaries.
`scratch/` is local and gitignored.

After clone, copy binaries by USB/rsync, then:
`slavv parity inspect-experiment-root`

Folder-purpose rules: [FOLDER_PURPOSE_GUIDE.md](../docs/reference/core/FOLDER_PURPOSE_GUIDE.md).

---

## Directory Structure

```
workspace/
├── oracles/          # Promoted MATLAB oracles (180709_E_full_v2, 180709_E_crop_M_v2)
├── runs/             # Live dests under oracle_180709_E/
├── reports/          # Archived proof JSON (phase1_volume_archive/)
├── datasets/         # Input volumes (content-addressed dirs)
├── scratch/          # Temporary files, logs, one-off scripts (gitignored)
└── index.jsonl       # Run index (auto-generated, gitignored)
```

### `oracles/` — MATLAB Truth Vectors

```
oracles/180709_E_full_v2/
├── 01_Input/
├── 03_Analysis/normalized/oracle/*.pkl
└── 99_Metadata/oracle_manifest.json
```

**How to create:** `slavv parity promote-oracle`

### `runs/` — Pipeline Experiment Runs

Live dests (freeze JSON `do_not_overwrite`; not the writer blocklist):
`canonical_full_v18`, `crop_M_exact_v3`, `crop_M_stretch_engine_v2`.

```
runs/oracle_180709_E/canonical_full_v18/
├── 02_Output/python_results/checkpoints/
├── 04_Edges/candidates.pkl
├── 03_Analysis/
└── 99_Metadata/
```

**Created by:** `slavv parity resume-exact-run` / `launch-exact-run` (or `slavv run --run-dir`).

### `reports/` — Archived Proof Summaries

Historical `prove-exact` JSON after multi-GB dests were removed:
`reports/phase1_volume_archive/`.

### `datasets/` — Test Volumes

Content-addressed dataset dirs with `01_Input/` TIFFs and
`99_Metadata/dataset_manifest.json`.

### `scratch/` — Temporary Files

One-off scripts, logs, MATLAB driver batches. Disposable. Not committed.

---

## When to Add Here

**Add here when:**
- ✅ Promoted Oracles, dataset TIFFs, or live dest checkpoints
- ✅ Archived proof JSON under `reports/`
- ✅ One-off exploration under `scratch/`

**Don't add here when:**
- ❌ Production code → `slavv_python/`
- ❌ Documentation → `docs/`
- ❌ Reusable test fixtures → `tests/support/`
- ❌ Dual-write `checkpoint_edge_candidates.pkl` / `chosen_edges.pkl` (gitignored fallbacks)

---

## Common Workflows

```powershell
slavv parity promote-oracle `
  --matlab-batch-dir D:\incoming\batch_260421-151654 `
  --oracle-root workspace\oracles\<oracle_id> `
  --dataset-file D:\datasets\volume.tif `
  --oracle-id <oracle_id>

slavv parity preflight-exact `
  --source-run-root workspace\runs\seed_run `
  --oracle-root workspace\oracles\<oracle_id> `
  --dest-run-root workspace\runs\my_current_code_trial

slavv parity prove-exact `
  --source-run-root workspace\runs\seed_run `
  --oracle-root workspace\oracles\<oracle_id> `
  --dest-run-root workspace\runs\my_current_code_trial `
  --stage all

slavv monitor --run-dir workspace\runs\my_current_code_trial
```

Do not overwrite live dests. `PROTECTED_DEST_NAMES` also blocks recreating
historical `canonical_full_v16`.

---

## Related Documentation

- [FOLDER_PURPOSE_GUIDE.md](../docs/reference/core/FOLDER_PURPOSE_GUIDE.md) — When to use each top-level folder
- [PARITY_PRE_GATE.md](../docs/reference/workflow/PARITY_PRE_GATE.md) — Parity experiment workflow
- [PARITY_CERTIFICATION_GUIDE.md](../docs/reference/workflow/PARITY_CERTIFICATION_GUIDE.md) — Certification commands
- [EXACT_PROOF_FINDINGS.md](../docs/reference/core/EXACT_PROOF_FINDINGS.md) — Live parity status
