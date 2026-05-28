---
description: "Run the full parity experiment workflow: capture candidates, compare against MATLAB oracle, diagnose gaps, and report match rate. Use after making parity-sensitive code changes."
name: "Run Parity Experiment"
argument-hint: "Optional: specific stage to test (energy, vertices, edges, network, all)"
agent: "agent"
---
Run the full parity experiment workflow to measure current MATLAB match rates.

## Prerequisites
- Oracle vectors must be promoted under `workspace/oracles/`
- A seed run must exist under `workspace/runs/`
- The parity experiment CLI must be available: `scripts/cli/parity_experiment.py`

## Workflow

### 1. Check Current Baseline
Read `docs/reference/core/EXACT_PROOF_FINDINGS.md` to understand the current match rate and known blockers.

### 2. Capture Fresh Candidates
```powershell
python scripts/cli/parity_experiment.py capture-candidates \
  --source-run-root workspace\runs\<seed_run> \
  --oracle-root workspace\oracles\<oracle_id> \
  --dest-run-root workspace\runs\<new_trial>
```

### 3. Run Exact Proof
```powershell
python scripts/cli/parity_experiment.py prove-exact \
  --source-run-root workspace\runs\<seed_run> \
  --oracle-root workspace\oracles\<oracle_id> \
  --dest-run-root workspace\runs\<new_trial> \
  --stage <stage_or_all>
```

### 4. Diagnose Gaps (if match rate regressed)
```powershell
python scripts/cli/parity_experiment.py diagnose-gaps \
  --run-root workspace\runs\<new_trial>
```

### 5. Report Results

Return:
1. **Match rate**: Current vs. previous (from EXACT_PROOF_FINDINGS.md)
2. **Stage breakdown**: Which stages passed/failed
3. **Gap analysis**: Top discrepant vertices/edges if applicable
4. **Regression check**: Whether changes improved, maintained, or regressed parity
5. **Recommended next step**: Based on docs/TODO.md checklist

## Guardrails
- Do not modify parity code during a measurement run.
- Preserve all trial outputs under `workspace/runs/` for reproducibility.
- Update `docs/reference/core/EXACT_PROOF_FINDINGS.md` if match rate changed significantly.
