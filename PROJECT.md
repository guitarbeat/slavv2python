# Project: slavv2python Parity
# Scope: Exact Parity Phase 1

## Architecture
- Module boundaries: Pipeline stages (Vertices, Edges, Network) tested against MATLAB exact parity truth.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | R1. Vertices Parity | `vertices` stage on crop harness | none | IN_PROGRESS |
| 2 | R2. Edges Parity | `edges` stage on crop harness | M1 | PLANNED |
| 3 | R3. Network Parity | `network` stage on crop harness | M2 | PLANNED |
| 4 | R4. Canonical Cert | Full exact proof on canonical volume | M3 | PLANNED |

## Interface Contracts
### Parity Harness ↔ Python Pipeline
- The Python pipeline must produce identical numeric output structurally matching the exact 1:1 MATLAB truth.
- Exact proof validation uses bitwise equality (not `np.isclose`) ensuring 0 extra and 0 missing across all vertices, edges, and network objects.
- All testing runs against standard crop harness: `180709_E_crop_M` for M1-M3.

## Code Layout
- Existing codebase layout applies. Core logic in `slavv_python/processing/stages/`.
