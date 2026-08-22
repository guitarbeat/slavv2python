---
name: slavv-parity-workflow
description: >-
  Standard operational protocol for running SLAVV parity preflight checks, cheap parity ladder verification, and exact proof evaluation under ADR 0011 and ADR 0012. Use when verifying MATLAB-to-Python parity, inspecting proof JSONs, or running preflight before launching a parity run.
---

# SLAVV Parity Workflow Protocol

## Overview
This skill defines the canonical operational workflow for executing and evaluating MATLAB-to-Python exact parity runs in `slavv2python`. It enforces the 3-tier verification sequence:
1. **Preflight Safety Inspection** (Lock checks, memory, parameter fingerprints)
2. **Cheap Parity Ladder Verification** (Unit tests -> crop harness -> full non-writer re-selection)
3. **Exact Proof Inspection** (ADR 0011 Energy/Vertices & ADR 0012 Edges/Network spatial multiset gates with mandatory `--require-evaluated` enforcement)

## Dependencies
- **`uv`**: For running Python environment test commands (`uv run pytest`).
- **`slavv` CLI**: Entrypoint for `slavv parity` commands (`inspect-proof`, `status-exact-run`, `preflight-exact`).

## Operational Sequence

### Tier 1: Parity Preflight Safety Check
Before starting, resuming, or launching any parity writer, perform the preflight safety audit:

```powershell
# 1. Verify environment and CLI entrypoint
.venv\Scripts\python.exe -m slavv_python info

# 2. Inspect active writer lease and PID lock
# Check if writer_lease.json exists in target run root.
# If active PID is running, FAIL LOUDLY. Do NOT launch concurrent writers.
```

**Preflight Failure Rules:**
- If `writer_lease.json` contains an active running PID, **BLOCK EXECUTION** and report the collision.
- If parameter fingerprints differ, **BLOCK EXECUTION** unless explicit diagnostic override is requested by the operator.

### Tier 2: Cheap Parity Ladder (Cheap-First Verification)
Before launching a full-volume writer or accepting code changes, run the cheap parity ladder:

```powershell
# Step 1: Focused Unit & Integration Tests
.venv\Scripts\python.exe -m pytest tests/unit/pipeline/test_global_watershed_comprehensive.py tests/integration/parity/test_parity_pre_gate_tier1.py

# Step 2: Crop Harness Frontier Diff Guard
.venv\Scripts\python.exe scripts/edges/frontier_diff.py `
  --run-dir workspace/runs/oracle_180709_E/crop_M_exact_v3 `
  --oracle-root workspace/oracles/180709_E_crop_M_v2 `
  --regenerate-python
```

### Tier 3: Proof Inspection & ADR 0012 Certification
Inspect stage proofs using the `slavv parity inspect-proof` surface. **Mandatory:** Always pass `--require-evaluated` to ensure strict spatial multiset evaluation:

```powershell
# Edges Stage Proof Inspection
slavv parity inspect-proof --path workspace/runs/oracle_180709_E/canonical_full_v18/03_Analysis/exact_proof_edges.json --require-evaluated

# Network Stage Proof Inspection
slavv parity inspect-proof --path workspace/runs/oracle_180709_E/canonical_full_v18/03_Analysis/exact_proof_network.json --require-evaluated
```

## Mandatory Parity Rules
1. **ONE TRUTH Authority:** `docs/reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk` is the SINGLE source of truth for live pass/fail status and claim roots.
2. **Never Overwrite Protected Baseline Dests:** Do NOT overwrite `canonical_full_v18`, `crop_M_exact_v3`, or `crop_M_stretch_engine_v2`.
3. **No Un-Evaluated Proof Claims:** Never claim Phase 1 closure or ADR 0012 pass unless `adr0012_evaluated: true` and `passed: true`.

## Common Pitfalls
- **Inventing KPIs outside ONE TRUTH:** Never cite outdated session diaries or raw candidate overlap % as ship gates.
- **Overwriting protected run roots:** Always create a new run destination directory when executing new parity runs.
- **Concurrent Writer Launches:** Running multiple parity writers on the same `--dest-run-root` will corrupt snapshot states.
