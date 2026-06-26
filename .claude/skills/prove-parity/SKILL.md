---
name: prove-parity
description: Run a SLAVV exact-parity proof for one pipeline stage (energy/vertices/edges/network) against the MATLAB oracle and summarize per-field pass/fail. Use when verifying or certifying MATLAB↔Python parity.
---

# prove-parity

Verify MATLAB↔Python parity for a SLAVV pipeline stage and report results honestly
(per-field pass/fail), using the spatial parity bars — not misleading overlap counts.

## Arguments
- `$1` = stage: `energy` | `vertices` | `edges` | `network` | `all` (default `all`).

## Fixed surfaces (crop iteration tier)
- Run root: `workspace/runs/oracle_180709_E/crop_M_exact`
- Oracle root: `workspace/oracles/180709_E_crop_M_v2` (batch `batch_260624-105705`)
- Checkpoints: `<run>/02_Output/python_results/checkpoints/checkpoint_<stage>.pkl`
- Use the repo `.venv` Python and set `PYTHONPATH` to the repo root.

## Steps
1. **Read status first.** Read `docs/reference/core/EXACT_PROOF_FINDINGS.md` for the
   stage's current state, known residuals, and ADR references (0011 float gate,
   0012 edge/network bars).
2. **Ensure the Python checkpoint exists.** If `checkpoint_<stage>.pkl` is missing:
   - For most stages, run the upstream writer/capture for that stage.
   - For **network** specifically, regenerate via *stage isolation* — feed the
     MATLAB curated edges + curated vertices so the network logic is tested, not
     upstream edge differences: load them with
     `load_normalized_matlab_stage(curated_edges*.mat, "edges")` and
     `..."vertices")`, build an `EdgeSet`/`VertexSet`, call
     `construct_network(...)`, and `joblib.dump(net.to_dict(), checkpoint_network.pkl)`.
     (See the `network-stage-parity` memory for the exact builder.)
3. **Run the proof** (dest = source so checkpoints are read in place):
   ```
   slavv parity prove-exact \
     --source-run-root workspace/runs/oracle_180709_E/crop_M_exact \
     --dest-run-root   workspace/runs/oracle_180709_E/crop_M_exact \
     --oracle-root     workspace/oracles/180709_E_crop_M_v2 \
     --stage <stage>
   ```
   Exit 0 = pass; exit 1 = a field failed (the command prints nothing itself).
4. **Read the report:** `<run>/03_Analysis/exact_proof.json` →
   `passed` and `stage_summaries[<stage>].first_failure` (field_path,
   mismatch_type, matlab/python previews). Summarize per-field.
5. **Interpret with the right bar:**
   - Continuous float fields certify under ADR 0011 `np.allclose(1e-7, 1e-9)`.
   - Edges: topology faithful; residual is watershed order-sensitivity — judge by
     voxel **ownership-map** agreement, not edge-pair overlap (ADR 0012).
   - Network: **topology exact** (endpoint-pair + bifurcation multisets); strand
     **geometry** under a **sub-voxel** trace tolerance (smoothing-kernel floor).

## Guardrails
- Do NOT report parity from raw edge-pair overlap — it can be inflated by
  coincidental wrong-grid matches.
- Use Git Bash `wc -l` (not PowerShell `Measure-Object`) for any line counts.
- Report failures faithfully with the actual `first_failure` evidence; never
  declare a stage green unless `passed: true` (or the residual is an explicitly
  documented sub-voxel/tolerance floor per ADR 0011/0012).
