# Parity Run Evidence Template

[Up: Reference Docs](../README.md) · [Exact Proof Findings](../core/EXACT_PROOF_FINDINGS.md) · [Parity Job Monitoring](PARITY_JOB_MONITORING.md)

Copy-paste checklist for reporting **writer completion** and **proof results** without duplicating narrative into [TODO.md](../../TODO.md).

**Rule:** [TODO.md](../../TODO.md) = checkboxes. [EXACT_PROOF_FINDINGS.md](../core/EXACT_PROOF_FINDINGS.md) = one ops-table row + short evidence block per event. This file = the shape of that block.

---

## Vocabulary

| Term | Meaning | Record when |
|------|---------|-------------|
| **Writer completed** | Pipeline stage finished; stage artifacts on disk | Energy/vertices/edges/network checkpoint or `best_*.npy` present |
| **Proof passed** | `prove-exact` reports strict zero for that stage | `exact_proof*.json` → `"passed": true` |
| **Proof failed** | Gate ran; mismatches remain | `exact_proof*.json` → `"passed": false` + mismatch diagnosis |

A finished writer does **not** imply proof passed.

---

## After every long writer

### 1. Status command

```powershell
slavv parity status-exact-run --run-dir workspace/runs/<run_root>
```

### 2. Artifact checklist

| Stage | Required artifacts |
|-------|-------------------|
| Energy | `02_Energy/best_energy.npy`, `02_Energy/best_scale.npy`, `02_Output/python_results/checkpoints/checkpoint_energy.pkl` |
| Vertices | `03_Vertices/` + `checkpoint_vertices.pkl` |
| Edges | `04_Edges/` + `checkpoint_edges.pkl` |
| Network | `05_Network/` + `checkpoint_network.pkl` |

### 3. Ops metadata (run-local)

| File | Purpose |
|------|---------|
| `99_Metadata/writer_lease.json` | Writer PID, `started_at`, terminal `status` |
| `99_Metadata/parity_job.json` | Detached job manifest (`started_at`, `ended_at`, `exit_code`, `status`) |
| `99_Metadata/parity_job.{out,err}.log` | Foreground/detached stdout/stderr |
| `99_Metadata/run_snapshot.json` | Stage progress (authoritative for pipeline state) |

### 4. Findings ops-table row (template)

```text
| <track> | workspace/runs/.../<run> | Writer: <stage> ✅ <ISO time> | Proof <stage>: ⏳ not run / ✅ pass / ❌ fail | Blocker: <one line> |
```

---

## After every `prove-exact` attempt

### Commands

```powershell
slavv parity prove-exact `
  --source-run-root workspace/runs/<run> `
  --dest-run-root workspace/runs/<run> `
  --oracle-root workspace/oracles/<oracle_id> `
  --stage <energy|vertices|edges|network|all>
```

### Proof artifacts (machine truth)

| File | Content |
|------|---------|
| `03_Analysis/exact_proof.json` | Full or sequence report |
| `03_Analysis/exact_proof_<stage>.json` | Per-stage report when run singly |
| `03_Analysis/exact_mismatch_<stage>.json` | Field-level mismatch counts + first coordinate |
| `03_Analysis/exact_mismatch_<stage>.txt` | Human-readable diagnosis |
| `03_Analysis/energy_probe_requests.json` | Adaptive probe ledger (energy failures only) |

### Evidence block (paste into findings)

```text
### <ISO date>: <run_id> — prove-exact --stage <stage>

- **Command:** prove-exact --stage <stage> on <run_root> vs <oracle_id>
- **Result:** PASS | FAIL
- **Writer artifacts:** <present | missing — list paths>
- **First failure:** <stage>.<field> @ <z,y,x> (from exact_proof*.json)
- **Mismatch counts:** (from exact_mismatch_<stage>.json)
  - <field>: <count> (max |Δ|=<value>)
- **Proof JSON:** 03_Analysis/exact_proof_<stage>.json
- **Next action:** <one line — no code speculation>
```

---

## Example — crop Energy 2026-06-22

```text
### 2026-06-22: crop_M_exact — prove-exact --stage energy

- **Command:** prove-exact --stage energy on crop_M_exact vs 180709_E_crop_M
- **Writer:** Energy completed 2026-06-22T18:24:34Z (~7h 44m, lattice 6000, n_jobs=1, lease PID 28520)
- **Result:** FAIL (strict np.equal)
- **Writer artifacts:** best_energy.npy, best_scale.npy present
- **First failure:** energy.energy @ (0,0,0) — scale 90 agrees; |ΔE|≈1e-14 (ULP)
- **Mismatch counts:**
  - energy: 3,823,893 (max |Δ|≈26.4) — ~3.8M with scale agreement are ULP-only
  - scale_indices: 19,412 (max |Δ|=72) — first @ (61,81,0) matlab 44 vs python 46
  - lumen_radius_microns: 8 (machine epsilon)
- **Proof JSON:** 03_Analysis/exact_proof_energy.json
- **Probe ledger:** 03_Analysis/energy_probe_requests.json (3650 groups)
- **Next action:** Triage scale-winner disagreements; do not refresh downstream crop stages
```

---

## After unit/regression tests (TODO only)

When a regression locks in a fix, check the matching box in [TODO.md](../../TODO.md) and cite the test path in findings **only if** it is evidence for an open proof blocker:

```text
- **Regression:** tests/unit/pipeline/energy/test_voxel_probe_regression.py (float64 bit-parity via tests/support/parity_harness.py)
```

Do not paste pytest summaries into findings.

---

## Promotion to `workspace/reports/`

Only after **strict-zero** `prove-exact-sequence` on the target surface (crop harness or canonical per [ADR 0009](../../adr/0009-parity-pre-gate-tiers.md)):

```powershell
Copy-Item workspace/runs/<run>/03_Analysis/exact_proof.json workspace/reports/<NAME>.json
```

See [PARITY_CERTIFICATION_GUIDE.md](PARITY_CERTIFICATION_GUIDE.md).