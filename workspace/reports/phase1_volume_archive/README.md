# Phase 1 volume archive (2026-08-18)

Proof JSON and run snapshots copied **before** deleting historical multi-GB dest
directories so the counted workspace could fit under 5 GB.

Live dests that still exist on disk (do **not** overwrite):

- `workspace/runs/oracle_180709_E/canonical_full_v18`
- `workspace/runs/oracle_180709_E/crop_M_exact_v3`
- `workspace/runs/oracle_180709_E/crop_M_stretch_engine_v2`
- `workspace/oracles/180709_E_full_v2`
- `workspace/oracles/180709_E_crop_M_v2`

Tracked hash bridge: `docs/reference/core/phase1-baseline-freeze.json`.

| Archived dest | Original path | Why removed |
|---|---|---|
| `canonical_full_v4` | `workspace/runs/oracle_180709_E/canonical_full_v4` | Energy/Vertices lineage proofs only; `v18` already has those checkpoints |
| `canonical_full_v5` | `workspace/runs/oracle_180709_E/canonical_full_v5` | Audit history |
| `canonical_full_v6` | `workspace/runs/oracle_180709_E/canonical_full_v6` | Audit history |
| `canonical_full_v7` | `workspace/runs/oracle_180709_E/canonical_full_v7` | Audit history |
| `canonical_full_v8` | `workspace/runs/oracle_180709_E/canonical_full_v8` | Audit history |
| `canonical_full_v10` | `workspace/runs/oracle_180709_E/canonical_full_v10` | Audit history |
| `canonical_full_v15` | `workspace/runs/oracle_180709_E/canonical_full_v15` | Audit history |
| `canonical_full_v16` | `workspace/runs/oracle_180709_E/canonical_full_v16` | Historical Network FAIL residual record; proofs only |
| `crop_M_exact` | `workspace/runs/oracle_180709_E/crop_M_exact` | Pre-v3 duplicate; live guard is `crop_M_exact_v3` |
| `crop_M_stretch_engine_v1` | `workspace/runs/oracle_180709_E/crop_M_stretch_engine_v1` | Stretch v1; live dest is `v2` |

`canonical_full_v4` Energy/Vertices proof SHA-256 values match the freeze JSON
(`exact_proof_energy.json` / `exact_proof_vertices.json`).

Cite live proofs with `slavv parity inspect-proof --path <json> --require-evaluated`
on `canonical_full_v18`. Do not resurrect these volumes.
