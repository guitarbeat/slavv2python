# Scope: R1. Vertices Parity

## Architecture
- Module boundaries: `slavv_python/processing/stages/vertices/` and exact parity verification.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | R1. Vertices Parity | `vertices` stage on crop harness (`180709_E_crop_M`) | none | PLANNED |

## Interface Contracts
- Must achieve strict 0 missing and 0 extra for the `vertices` stage on `180709_E_crop_M`.
- Must run exact proof sequentially against MATLAB exact parity truth.
- Must strictly follow the MATLAB 1:1 structure and use exact bitwise equality instead of `np.isclose`.

## Testing Command
`python scripts/cli/parity_experiment.py prove-exact-sequence --source-run-root workspace/runs/oracle_180709_E/crop_M_exact --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact --oracle-root workspace/oracles/180709_E_crop_M`
(Note: Only check the vertices stage output. If you need to test just the vertices stage, you may be able to use `prove-exact` targeting just vertices if it's faster for iteration, but the final acceptance is via sequence).
