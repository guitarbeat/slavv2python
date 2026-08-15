# Exact Proof Findings

[Up: Reference Docs](../README.md) · [Authority map](../../README.md#documentation-authority-map-one-concept--one-home) · [HANDOFF](../../../.claude/HANDOFF.md) · [TODO](../../TODO.md)

**Last Updated:** 2026-08-14  
**Role:** **Only** live source of truth for exact-route MATLAB↔Python parity status (runs, proofs, blockers, residual claim).  
**Not here:** task checkboxes ([TODO](../../TODO.md)), operator commands ([HANDOFF](../../../.claude/HANDOFF.md)), figure paint constants ([parity_campaign_series.py](../../../figures/parity_campaign_series.py) — mirror KPIs only), investigation diary ([archive](../../investigations/exact-proof-findings-diary/README.md)).

---

## ONE TRUTH — Phase 1 parity (validated from disk)

> **Answer:** Phase 1 exact-route **Certification is CLOSED** on full `180709_E`.  
> Energy, Vertices, Edges, and Network all pass their certification bars on the claim surface. The former Network one-strand gap was an Edge Selection Ranking Residual (original-field traces vs claimed `energy_map`); baking Claimed Trace Energy at watershed finalize ([ADR 0013](../../adr/0013-claimed-energy-trace-provenance.md)) closes it on `canonical_full_v18`.

| Stage | Verdict | Claim surface / evidence | Notes |
| :--- | :--- | :--- | :--- |
| **Energy** | ✅ **PASS** (ADR 0011) | Full-volume proof lineage: `canonical_full_v4` `03_Analysis/exact_proof_energy.json` (`passed: true`). Seeded into later claim roots. | Discrete scale indices exact; continuous under `np.allclose`. |
| **Vertices** | ✅ **PASS** (ADR 0011) | `canonical_full_v4` `exact_proof_vertices.json` (`passed: true`). | Positions/scales exact. |
| **Edges** | ✅ **PASS** (ADR 0012 evaluated) | **`canonical_full_v18`** `03_Analysis/exact_proof_edges.json` | Connections **69,500 / 69,500**; ownership **5,843,205 / 5,843,213** (**99.999863%**); trace failures **0** / 69,500; `adr0012_evaluated: true`. Final Edge Set keeps oracle hub pair `(34897, 38584)` and drops residual extra `(26444, 38584)`. |
| **Network** | ✅ **PASS** (ADR 0012 evaluated) | **`canonical_full_v18`** `03_Analysis/exact_proof_network.json` | Strand endpoint-pair + bifurcation multisets match (`network_adr0012_gate.adr0012_evaluated: true`, `passed: true`). Strand count **48,049 / 48,049**. |

**Oracle:** `workspace/oracles/180709_E_full_v2` (batch `batch_260626-125646`).  
**Claim run root:** `workspace/runs/oracle_180709_E/canonical_full_v18`.  
**Historical claim (open residual record):** `canonical_full_v16` — Edges ✅ Network ❌ (one strand); preserve in place; do not overwrite.  
**Phase 1 closes when** evaluated Edges **and** Network both pass on a fresh full claim root — **met on `v18`**. Cite those JSON files with `slavv parity inspect-proof --path <json> --require-evaluated`.

### Disk revalidation stamp

**2026-08-14** — re-read / re-proved on disk:

- `canonical_full_v18` `exact_proof_edges.json`: `passed=true`, `edges_adr0012_gate.adr0012_evaluated=true`, connections **69,500 / 69,500**, ownership **0.9999986308902311**, `trace_n_failures=0`.
- `canonical_full_v18` `exact_proof_network.json`: `passed=true`, `network_adr0012_gate.adr0012_evaluated=true`, strand pairs **48,049 / 48,049**.
- Final edges on `v18`: has oracle `(34897, 38584)`, lacks extra `(26444, 38584)`. Contrast `v16` finals: had extra, lacked oracle.
- `v18` candidate traces for residual hub: claimed maxes **0.0** (extra) and **−0.238…** (oracle) — match MATLAB L846 claimed map; `v16` candidates still show original-field **−9.24 / −7.73**.
- Energy/Vertices stage JSON live under `canonical_full_v4` (lineage seed); carried into `v18`.

### Former residual (closed on v18)

- **Crop guard closed:** `crop_M_exact_v3` re-selection undirected pair overlap **15,511 / 15,511** vs `180709_E_crop_M_v2`. On-disk crop `prove-exact --stage edges` may still be unevaluated (missing ownership map). Trust the pair-set / re-selection check.
- **Raw Candidate Sets already match.** MATLAB raw dumps and Python candidates are the **same undirected pairs**. MATLAB **does** emit `(26444, 38584)`.
- **Mechanism:** after resample, extra `(26444, 38584)` and oracle `(34897, 38584)` tied on `max`. Degree-excess kept the **earlier** row. Python `v16` stored traces sampled the **original** energy field (extra looked better). MATLAB `sort_edges` ranks `max` of the **claimed/penalized** `energy_map` (L445 write, L846 sample): extra `0.0`, oracle `−0.239`, so extra sat last and was dropped.
- **Production fix (landed):** sample `claim_map.energy_map` for watershed energy traces ([ADR 0013](../../adr/0013-claimed-energy-trace-provenance.md)) + MATLAB `sort_edges` (raw `max`, ascend) **before** resampled `clean_edge_pairs`. Verified on `canonical_full_v18` Edge Set + Network.
- **Do not claim from `canonical_full_v17`:** contaminated; deleted 2026-08-13.
- Cheap tests: `tests/unit/pipeline/test_watershed_energy_map_sort_experiments.py`. Compare/cite through `slavv_python.analytics.parity.experiments`. Runbook: [raw-vs-final-candidate-compare.md](../../solutions/parity/raw-vs-final-candidate-compare.md).

**Archived (do not treat as live residual):** join-emission attempts A–C, the `find(...,'last')` rewrite conclusion, and the session diary live in [exact-proof-findings-diary](../../investigations/exact-proof-findings-diary/README.md).

**Figure KPI mirror:** update [`figures/parity_campaign_series.py`](../../../figures/parity_campaign_series.py) only when the table above moves; then regenerate claim figures.

**Spec:** [phase-1-exact-route-spec.md](../../plans/phase-1-exact-route-spec.md)

---

## True zero-tolerance stretch (separate from Phase 1)

> **Phase 1 Certification remains CLOSED** on `canonical_full_v18` (see [ONE TRUTH](#one-truth--phase-1-parity-validated-from-disk)). Stretch greens/reds **never** rewrite that answer or ADR 0011/0012 ship bars.

This subsection tracks the post–Phase 1 **true zero-tolerance** program (bit-equal Energy floats + discrete strict fields under `--strict-floats`). Plan: [2026-08-14-004-feat-true-zero-tolerance-parity-stretch-plan.md](../../plans/2026-08-14-004-feat-true-zero-tolerance-parity-stretch-plan.md). Helpers: `slavv_python.analytics.parity.proof.stretch`.

| Concept | Rule |
| :--- | :--- |
| Compare gate | `prove-exact --strict-floats` only; default allclose is **not** stretch success |
| Crop → full | Hard unlock token (`stretch_crop_unlock.json`) scoped by field set (`energy` vs `energy+discrete`) |
| Status taxonomy | `blocked_float_path` / `incomplete_discrete` / `incomplete_infra` / `incomplete_at_full` / `stretch_complete` (Energy **and** discrete at full) |
| Dest roots | New stretch run roots only — never overwrite `canonical_full_v18` |

Live stretch status is written beside stretch run artifacts (`stretch_status.json`), not into ONE TRUTH.

**Session status (2026-08-14):** Foundation landed (unlock/status, engine adapter, origin dispatch, gates). Crop Energy bit-equal is **`incomplete_infra`** on the repo Python 3.12 venv — MATLAB R2019a Engine for Python supports 2.7/3.5/3.6/3.7 only. Not redefined as ADR 0011 allclose success.

---

## Audit inventory (folders, not a second verdict)

Pass/fail is only in [ONE TRUTH](#one-truth--phase-1-parity-validated-from-disk). This table says which folders exist and what they are for.

| Class | Path | Role |
|-------|------|------|
| Live oracle (full) | `workspace/oracles/180709_E_full_v2` | Proofs only (`batch_260626-125646`) |
| Live oracle (crop) | `workspace/oracles/180709_E_crop_M_v2` | Crop proofs (`batch_260624-105705`) |
| **Claim run (closed)** | `workspace/runs/oracle_180709_E/canonical_full_v18` | Phase 1 claim surface — Edges + Network evaluated PASS |
| Historical claim | `canonical_full_v16` | Pre-ranking-fix residual record (Network FAIL); preserve |
| Lineage seed | `canonical_full_v4` Energy/Vertices; `v8` energy `.npy` | Seed successors |
| Crop guard | `crop_M_exact_v3` candidates | Regression; do not cite unevaluated proof JSON |
| Audit history | `v5`–`v15` completed writers | Keep |
| Removed | `canonical_full_v17` | Contaminated; deleted 2026-08-13 |

Evidence template: [PARITY_RUN_EVIDENCE.md](../workflow/PARITY_RUN_EVIDENCE.md)

---

## Active blockers

1. **None for Phase 1 Closure** — evaluated Edges and Network both PASS on `canonical_full_v18`.
2. **Strict-field stretch** (optional) — exact `connections` / order-sensitive emission remains non-blocking on crop.
3. **Crop / frontier / cleanup** — regression guards only (closed).

**Superseded:** “100% parity” without evaluated proofs, “>95% match”, “block on 80% crop overlap”, “crop one-pair swap is the open loop”, strict-field fallback as closure, join-emission / tie-scan as the ship-gate change, Network rewrite as the default residual response.

---

## Cold-start protocol

1. Read **[ONE TRUTH](#one-truth--phase-1-parity-validated-from-disk)**. Do **not** use the [diary archive](../../investigations/exact-proof-findings-diary/README.md) as status.
2. Read **[.claude/HANDOFF.md](../../../.claude/HANDOFF.md)** for commands only.
3. No concurrent writer: read `99_Metadata/writer_lease.json` and test that PID. Do **not** block on `slavv jobs list` (hangs).
4. `slavv parity ensure-oracle-artifacts --oracle-root workspace/oracles/180709_E_crop_M_v2 --stage all --no-repair` (and the same for `180709_E_full_v2` before canonical work).
5. Residual / regression: cheap [Parity Experiment](../../../AGENTS.md#parity-experiment) first (`slavv_python.analytics.parity.experiments`). Crop is a regression guard.
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
| Claimed energy trace provenance (ADR 0013) | [0013-claimed-energy-trace-provenance.md](../../adr/0013-claimed-energy-trace-provenance.md) |
| Parity experiment hygiene | [parity-experiment-hygiene.md](../../solutions/best-practices/parity-experiment-hygiene.md) |
| Curated vertices rank-ramp energies | [curated-vertices-rank-ramp-energies.md](../../solutions/integration-issues/curated-vertices-rank-ramp-energies.md) |

_Add rows here when a new compound doc is parity-relevant; do not duplicate full write-ups in this file._
