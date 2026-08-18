# Workspace

See [Experiment Root](../AGENTS.md#experiment-root).

**Purpose:** On-disk Oracles, datasets, live Claim Run Root / crop guard /
stretch dest, and archived proofs. GitHub does not carry those binaries.
`scratch/` is local and gitignored.

After clone, copy binaries by USB/rsync, then:
`slavv parity inspect-experiment-root`

---

## Directory Structure

```
workspace/
├── oracles/          # Preserved MATLAB truth vectors for parity testing
├── runs/             # Trial pipeline runs with checkpoints and outputs
├── reports/          # Promoted proof summaries
├── datasets/         # Test datasets and sample volumes
├── scratch/          # Temporary files, logs, one-off scripts
└── index.jsonl       # Run index (auto-generated)
```

---

## What Lives Here

### `oracles/` — MATLAB Truth Vectors
Preserved MATLAB oracle vectors for exact parity comparison:

```
oracles/
└── 180709_crop_M/
    ├── energy_oracle.mat
    ├── vertices_oracle.mat
    ├── edges_oracle.mat
    ├── network_oracle.mat
    └── oracle_metadata.json
```

**How to create:** Use `scripts/parity_experiment.py promote-oracle` to convert MATLAB batch output into a reusable oracle.

### `runs/` — Pipeline Experiment Runs
Trial pipeline runs with full checkpoints and outputs:

```
runs/
└── crop_M_exact/
    ├── _slavv_run/           # Run metadata and state
    ├── checkpoints/          # Stage checkpoints
    ├── artifacts/            # Diagnostic outputs
    └── comparison_results/   # Parity diff reports
```

**Created by:** `slavv run --run-dir workspace\runs\my_experiment`

### `reports/` — Promoted Proof Summaries
Hand-curated summaries of important parity proof results:

```
reports/
└── crop_M_exact_proof_2024-06-09.md
```

**When to add:** When a parity run produces noteworthy results worth preserving in narrative form.

### `datasets/` — Test Volumes
Test datasets and sample volumes for experiments:

```
datasets/
├── crop_M_for_testing.tif
├── synthetic_small.tif
└── 180709_E_full.tif  (symlink or copy)
```

**Note:** Large datasets should be stored elsewhere and symlinked here when possible.

### `scratch/` — Temporary Files
One-off exploration scripts, debug logs, quick checks:

```
scratch/
├── debug_edge_issue.py
├── quick_comparison.log
└── temp_analysis.ipynb
```

**Rule:** Anything in `scratch/` is disposable. Don't put production-quality code here.

---

## When to Add Here

**Add here when:**
- ✅ It's experiment output
- ✅ It's large binary data (oracles, volumes, checkpoints)
- ✅ It's temporary or frequently changing
- ✅ It's specific to your local development
- ✅ It's a one-off exploration script

**Don't add here when:**
- ❌ Other developers need it → Put in `slavv_python/`, `tests/`, or `scripts/`
- ❌ It's documentation → Put in `docs/`
- ❌ It's a reusable test fixture → Put in `tests/support/`
- ❌ It's production code → Put in `slavv_python/`

---

## Common Workflows

### Creating an Oracle
```powershell
# 1. Run MATLAB vectorization (outside this repo)
# 2. Promote output to oracle
python scripts/parity_experiment.py promote-oracle \
  --matlab-batch-dir D:\incoming\batch_260421 \
  --oracle-root workspace\oracles\crop_M \
  --dataset-file D:\datasets\crop_M.tif \
  --oracle-id crop_M
```

### Running a Parity Experiment
```powershell
# 1. Run preflight check
python scripts/parity_experiment.py preflight-exact \
  --source-run-root workspace\runs\seed_run \
  --oracle-root workspace\oracles\crop_M \
  --dest-run-root workspace\runs\my_trial

# 2. Run full exact proof
python scripts/parity_experiment.py prove-exact \
  --source-run-root workspace\runs\seed_run \
  --oracle-root workspace\oracles\crop_M \
  --dest-run-root workspace\runs\my_trial \
  --stage all

# 3. Monitor progress
slavv monitor --run-dir workspace\runs\my_trial
```

### Quick Exploration
```powershell
# Create a throwaway script in scratch/
python workspace/scratch/test_new_idea.py
```

---

## .gitignore

`workspace/scratch/` and dual-write leftovers (`checkpoint_edge_candidates.pkl`,
`chosen_edges.pkl`) stay ignored. Oracle / dest / dataset binaries stay local
(not GitHub LFS). See [Experiment Root](../AGENTS.md#experiment-root).

## File Size Considerations

Do not add new full-volume writers to git. Copy Experiment Root binaries by
USB/rsync. Completeness is `slavv parity inspect-experiment-root`.

**Storage recommendations:**
- Keep `workspace/` on a local SSD for performance
- New overnight writers stay untracked until promoted
- Scratch stays local

---

## Related Documentation

- [FOLDER_PURPOSE_GUIDE.md](../docs/reference/core/FOLDER_PURPOSE_GUIDE.md) — When to use each top-level folder
- [PARITY_PRE_GATE.md](../docs/reference/workflow/PARITY_PRE_GATE.md) — Parity experiment workflow
- [PARITY_CERTIFICATION_GUIDE.md](../docs/reference/workflow/PARITY_CERTIFICATION_GUIDE.md) — Certification commands
- [EXACT_PROOF_FINDINGS.md](../docs/reference/core/EXACT_PROOF_FINDINGS.md) — Live parity status
