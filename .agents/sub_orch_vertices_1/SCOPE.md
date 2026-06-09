# Scope: R1. Vertices Parity

## Architecture
- Module boundaries: Pipeline stages (Vertices) tested against MATLAB exact parity truth.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Vertices Parity | `vertices` stage on crop harness | none | IN_PROGRESS |

## Interface Contracts
### Parity Harness ↔ Python Pipeline
- The Python pipeline must produce identical numeric output structurally matching the exact 1:1 MATLAB truth.
- Exact proof validation uses bitwise equality (not `np.isclose`) ensuring 0 extra and 0 missing across all vertices.
- Testing runs against standard crop harness: `180709_E_crop_M`.
- dest-run-root: workspace/runs/oracle_180709_E/crop_M_exact
- oracle-root: workspace/oracles/180709_E_crop_M
