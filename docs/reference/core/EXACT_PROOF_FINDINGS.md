# Exact Proof Findings

[Up: Reference Docs](../README.md) · [Authority map](../../README.md#documentation-authority-map-one-concept--one-home) · [HANDOFF](../../../.claude/HANDOFF.md) · [TODO](../../TODO.md)

**Last Updated:** 2026-08-13  
**Role:** **Only** live source of truth for exact-route MATLAB↔Python parity status (runs, proofs, blockers, residual claim).  
**Not here:** task checkboxes ([TODO](../../TODO.md)), operator commands ([HANDOFF](../../../.claude/HANDOFF.md)), figure paint constants ([parity_campaign_series.py](../../../figures/parity_campaign_series.py) — mirror KPIs only), investigation diary ([archive](../../investigations/exact-proof-findings-diary/README.md)).

---

## ONE TRUTH — Phase 1 parity (validated from disk)

> **Answer:** We do **not** have 100% end-to-end MATLAB≡Python certification.  
> **Phase 1 is OPEN.** Three of four stages pass their certification bars on the claim surface; Network fails ADR 0012 multiset equality by **one strand**.

| Stage | Verdict | Claim surface / evidence | Notes |
| :--- | :--- | :--- | :--- |
| **Energy** | ✅ **PASS** (ADR 0011) | Full-volume proof lineage: `canonical_full_v4` `03_Analysis/exact_proof_energy.json` (`passed: true`). Seeded into later claim roots. | Discrete scale indices exact; continuous under `np.allclose`. |
| **Vertices** | ✅ **PASS** (ADR 0011) | `canonical_full_v4` `exact_proof_vertices.json` (`passed: true`). | Positions/scales exact. |
| **Edges** | ✅ **PASS** (ADR 0012 evaluated) | **`canonical_full_v16`** `03_Analysis/exact_proof_edges.json` | Connections **69,500 / 69,500**; ownership **5,843,205 / 5,843,213** (**99.999863%**); trace failures **0** / 69,499; `adr0012_evaluated: true`. |
| **Network** | ❌ **FAIL** (ADR 0012) | **`canonical_full_v16`** `03_Analysis/exact_proof_network.json` | Strand endpoint-pair multiset: Python **48,048** vs MATLAB **48,049**. `release_evidence.json` `proof_passed: false`. **Open ship gate.** |

**Oracle:** `workspace/oracles/180709_E_full_v2` (batch `batch_260626-125646`).  
**Claim run root:** `workspace/runs/oracle_180709_E/canonical_full_v16`.  
**Phase 1 closes only when** evaluated Edges **and** Network both pass on a fresh full claim root (see [ADR 0012](../../adr/0012-edge-watershed-parity-bar.md)). Cite those JSON files with `slavv parity inspect-proof --path <json> --require-evaluated`.

### Disk revalidation stamp

**2026-07-16** — re-read JSON on disk (no re-run). Confirmed:

- `exact_proof_edges.json`: `passed=true`, `edges_adr0012_gate.adr0012_evaluated=true`, `n_python_connections=n_matlab_connections=69500`, `ownership_map_agreement_rate=0.9999986308902311`, `trace_n_failures=0`.
- `exact_proof_network.json`: `passed=false`, first failure `network.strands` strand endpoint-pair multiset mismatch, shapes `[48048,2]` vs `[48049,2]`.
- Energy/Vertices stage JSON live under `canonical_full_v4` (lineage seed); not re-proved on `v16` path because those stages were carried forward unchanged.

### Active residual (why Network is red)

- **Crop guard closed:** `crop_M_exact_v3` re-selection undirected pair overlap **15,511 / 15,511** vs `180709_E_crop_M_v2`. On-disk `prove-exact --stage edges` there is **not** evaluated (`adr0012_evaluated: false`, missing Python ownership map). Trust the pair-set / re-selection check, not that JSON as a spatial-bar verdict.
- **Raw Candidate Sets already match.** MATLAB `workspace/scratch/matlab_edge_dump/raw_full_candidates.mat` and Python v16/v17 candidates are the **same 84,650 undirected pairs** (`only_py=0`, `only_mat=0`). MATLAB **does** emit `(26444, 38584)`. The 15,150 “Python-only extras vs MATLAB finals” are extras vs the **cleaned** 69,500 — both languages’ raw sets contain them.
- **Full residual:** after resample, extra `(26444, 38584)` and oracle `(34897, 38584)` tie on `max` (`−4.870…`). Degree-excess keeps the **earlier** row. Python’s stored traces sample the **original** energy field (extra looks better, `−9.24` vs `−7.73`). MATLAB `sort_edges` ranks `max` of the **claimed/penalized** `energy_map` (L445 write, L846 sample): extra `0.0`, oracle `−0.239`, so extra sits last and is dropped.
- **Ablation:** drop only `cand 46698` still yields 69,500 / 69,500 — that fakes MATLAB’s post-`sort_edges` rank; it does **not** prove MATLAB never emitted the pair.
- Production fix = sample `claim_map.energy_map` for watershed energy traces + MATLAB `sort_edges` (raw `max`, ascend) **before** resampled `clean_edge_pairs`. **Not** join-emission / tie-scan / cleanup secondary keys.
- **Do not claim from `canonical_full_v17`:** mixed Energy rerun, missing local energy/vertices checkpoints, stale “running” snapshot, job “succeeded” on permission denied. Deleted 2026-08-13. New Edges→Network successor only.
- Scratch: `raw_full_candidates.mat`, `global_presort_candidates.mat`, `full_residual_pair_raw.npz`. Cheap tests: `tests/unit/pipeline/test_watershed_energy_map_sort_experiments.py`. Compare/cite through `slavv_python.analytics.parity.experiments`. Runbook: [raw-vs-final-candidate-compare.md](../../solutions/parity/raw-vs-final-candidate-compare.md).

**Archived (do not treat as live residual):** join-emission attempts A–C, the `find(...,'last')` rewrite conclusion, “fix deployed / closure run in progress,” and the session diary live in [exact-proof-findings-diary](../../investigations/exact-proof-findings-diary/README.md).

**Figure KPI mirror:** update [`figures/parity_campaign_series.py`](../../../figures/parity_campaign_series.py) only when the table above moves; then regenerate claim figures.

**Spec:** [phase-1-exact-route-spec.md](../../plans/phase-1-exact-route-spec.md)

---

## Audit inventory (folders, not a second verdict)

Pass/fail is only in [ONE TRUTH](#one-truth--phase-1-parity-validated-from-disk). This table says which folders exist and what they are for.

| Class | Path | Role |
|-------|------|------|
| Live oracle (full) | `workspace/oracles/180709_E_full_v2` | Proofs only (`batch_260626-125646`) |
| Live oracle (crop) | `workspace/oracles/180709_E_crop_M_v2` | Crop proofs (`batch_260624-105705`) |
| Claim run | `workspace/runs/oracle_180709_E/canonical_full_v16` | Live residual / Network ship gate |
| Lineage seed | `canonical_full_v4` Energy/Vertices; `v8` energy `.npy` | Seed successors |
| Crop guard | `crop_M_exact_v3` candidates | Regression; do not cite unevaluated proof JSON |
| Audit history | `v5`–`v15` completed writers | Keep |
| Removed | `canonical_full_v17` | Contaminated; deleted 2026-08-13 |
| Successor writer | `canonical_full_v18` | Edges→Network started 2026-08-13 (`resume-exact-run`, not `launch-exact-run`). **Not** the claim root until evaluated proofs pass. |

Evidence template: [PARITY_RUN_EVIDENCE.md](../workflow/PARITY_RUN_EVIDENCE.md)

---

## Active blockers

1. **Full Edge Set residual** — [ONE TRUTH residual](#active-residual-why-network-is-red). Not a Network rewrite; not cleanup reorder.
2. **Phase 1 ship gate = Network ADR 0012 multiset** — Edges evaluated PASS on claim root; Network FAIL until Edge Set multiset matches (MATLAB-edge isolation exact).
3. **Crop / frontier / cleanup** — regression guards only (closed).

**Superseded:** “100% parity”, “>95% match”, “block on 80% crop overlap”, “crop one-pair swap is the open loop”, strict-field fallback as closure, join-emission / tie-scan as the ship-gate change.

---

## Cold-start protocol

1. Read **[ONE TRUTH](#one-truth--phase-1-parity-validated-from-disk)**. Do **not** use the [diary archive](../../investigations/exact-proof-findings-diary/README.md) as status.
2. Read **[.claude/HANDOFF.md](../../../.claude/HANDOFF.md)** for commands only.
3. No concurrent writer: read `99_Metadata/writer_lease.json` and test that PID. Do **not** block on `slavv jobs list` (hangs).
4. `slavv parity ensure-oracle-artifacts --oracle-root workspace/oracles/180709_E_crop_M_v2 --stage all --no-repair` (and the same for `180709_E_full_v2` before canonical work).
5. Residual loop: cheap [Parity Experiment](../../../AGENTS.md#parity-experiment) first (`slavv_python.analytics.parity.experiments`). Crop is a regression guard.
6. Cite proofs with `slavv parity inspect-proof --path <json> --require-evaluated`.
7. Capture evidence per [PARITY_RUN_EVIDENCE.md](../workflow/PARITY_RUN_EVIDENCE.md). Re-synthesize HANDOFF if this file’s ONE TRUTH section changes.

---

## Compound learnings (parity-related)

Curated index of solved problems under `docs/solutions/`. Search via YAML frontmatter; see [docs/solutions/README.md](../../solutions/README.md).

| Topic | Doc |
|-------|-----|
| MATLAB energy HDF5 + `promote-oracle` | [matlab-v200-energy-hdf5-oracle-loader.md](../../solutions/integration-issues/matlab-v200-energy-hdf5-oracle-loader.md) |
| Detached exact-run jobs (Windows: use `Start-Process resume-exact-run`, not `launch-exact-run`) | [detached-exact-run-jobs.md](../../solutions/parity/detached-exact-run-jobs.md) |
| Run/proof evidence template | [PARITY_RUN_EVIDENCE.md](../workflow/PARITY_RUN_EVIDENCE.md) |
| Sparse Meshgrid Memory Optimization | [sparse-meshgrid-memory-optimization.md](../../solutions/parity/sparse-meshgrid-memory-optimization.md) |
| MATLAB Stride Phase Lead | [matlab-stride-phase-lead.md](../../solutions/parity/matlab-stride-phase-lead.md) |
| Vertex NMS structuring element (float radii) | [vertex-structuring-element-float-radius.md](../../solutions/parity/vertex-structuring-element-float-radius.md) |
| Edge watershed faithfulness (seeds=2, no conflict painting) | [edge-watershed-matlab-faithfulness.md](../../solutions/parity/edge-watershed-matlab-faithfulness.md) |
| Raw vs final candidate compare (residual class) | [raw-vs-final-candidate-compare.md](../../solutions/parity/raw-vs-final-candidate-compare.md) |
| Parity experiment hygiene | [parity-experiment-hygiene.md](../../solutions/best-practices/parity-experiment-hygiene.md) |
| Curated vertices rank-ramp energies | [curated-vertices-rank-ramp-energies.md](../../solutions/integration-issues/curated-vertices-rank-ramp-energies.md) |

_Add rows here when a new compound doc is parity-relevant; do not duplicate full write-ups in this file._
