# Stage-Isolated Network Parity 2026-04-07

This note captures a stage-isolated parity probe motivated by the repo docs:
if there is a faster path to convergence, we should prove later stages
independently before continuing to tune the frontier tracer.

## Question

If Python is given exact MATLAB `energy`, exact MATLAB `vertices`, and exact
MATLAB `edges`, does the Python `network` stage already converge to MATLAB?

## Why This Matters

If the answer is yes, then the shortest path to full parity is:

1. hold `network` fixed as already solved on the parity surface,
2. focus all remaining algorithm work on `edges`,
3. only rerun `network` as a downstream confirmation step.

That is a much better convergence loop than treating `edges` and `network` as
one combined failing surface.

## Experiment Setup

- MATLAB batch:
  `comparisons/20260406_conflict_provenance_trial/01_Input/matlab_results/batch_260406-164522`
- Probe workspace:
  `workspace/tmp_network_stage_probe/`
- Authoritative comparison function:
  `slavv.evaluation.metrics.compare_results(...)`

The probe variants were:

- `imported_all`
  - import MATLAB `energy`, `vertices`, `edges`, and `network`
  - expected to match, since both sides are MATLAB-imported
- `network_rerun`
  - import MATLAB `energy`, `vertices`, `edges`
  - rerun Python from `network`
- `network_rerun_exact`
  - same as `network_rerun`, but replay normalized comparison params from
    `99_Metadata/comparison_params.normalized.json`
- `network_rerun_exactflag`
  - same as `network_rerun_exact`, but explicitly set
    `comparison_exact_network = True` before rerunning `network`

## Results

| Probe | Vertices Exact | Edges Exact | Network Exact | MATLAB Strands | Python Strands | Passed |
| --- | --- | --- | --- | ---: | ---: | --- |
| `imported_all` | `true` | `true` | `true` | `682` | `682` | `true` |
| `network_rerun` | `true` | `true` | `false` | `682` | `364` | `false` |
| `network_rerun_exact` | `true` | `true` | `false` | `682` | `364` | `false` |
| `network_rerun_exactflag` | `true` | `true` | `true` | `682` | `682` | `true` |

## Main Finding

Yes: Python `network` assembly already reaches exact parity when fed exact
MATLAB `edges`, but only when `comparison_exact_network=True`.

That means the current dominant blocker is not generic graph assembly. The
remaining convergence work should focus on getting Python `edges` to match the
MATLAB edge set, while continuing to use parity-mode `network` assembly as the
downstream check.

## Important Operational Nuance

The normalized comparison params file in `99_Metadata/` did not preserve the
effective parity behavior by itself for this replay. Reusing those params
without explicitly forcing `comparison_exact_network=True` produced a false
negative stage-isolated result (`364` strands instead of `682`).

Implication:

- stage-isolated reruns need to set parity-mode `network` behavior explicitly,
  or
- the comparison workflow should persist that effective flag in a replay-safe
  way.

Follow-up:

- the comparison CLI now supports this directly through
  `--python-parity-rerun-from network`
- comparison-mode Python execution now forces
  `comparison_exact_network=True`
- normalized comparison params snapshots now record both
  `comparison_exact_network` and `python_parity_rerun_from`

CLI validation:

- run root:
  `workspace/tmp_network_stage_probe/cli_probe`
- command:
  `python workspace/scripts/cli/compare_matlab_python.py --input <repo>/data/slavv_test_volume.tif --output-dir <repo>/workspace/tmp_network_stage_probe/cli_probe --skip-matlab --python-parity-rerun-from network --comparison-depth deep`
- observed result:
  - MATLAB vertices/edges/strands: `1682 / 1379 / 682`
  - Python vertices/edges/strands: `1682 / 1379 / 682`
  - parity gate: `PASS`

## Conclusion

The better path to convergence is stage-isolated:

- prove `network` independently with imported MATLAB `edges`,
- keep `network` out of the main suspect list unless this probe regresses,
- concentrate parity debugging on frontier candidate generation and edge
  selection.
