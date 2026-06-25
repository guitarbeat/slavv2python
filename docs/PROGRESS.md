# SLAVV milestone background

**Last synthesized:** 2026-06-25

This page is a stable orientation summary, not a live dashboard. Current tasks
are in [TODO.md](TODO.md); verified parity runs, proofs, and blockers are in
[EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md).

## Current milestone

Phase 1 seeks strict sequential exact-route certification on full `180709_E`:
Energy → Vertices → Edges → Network. The crop harness is the required pre-gate.

| Stage | Current certification status |
|---|---|
| Energy | CERTIFIED (crop v2, ADR 0011 gate; batch_260624-105705): scale winners strict 0; `energy.energy` passes `np.allclose` (rtol=1e-7/atol=1e-9, max \|Δ\|≈2×10⁻¹¹). |
| Vertices | CERTIFIED (crop v2): positions + scales exact (13,706 = 13,706); energies pass under the ADR 0011 `np.allclose` gate. |
| Edges | In progress (not yet certified); prior informal pair-match baselines are not certification. |
| Network | Pending upstream proof. |

## Durable progress

- The exact route uses `[Y, X, Z]` internal alignment and Fortran-order
  tie-breaking to reproduce MATLAB behavior.
- Exact Energy processing uses an incremental best-scale engine to avoid the
  large 4D per-chunk Energy stack.
- Pipeline policy, lattice rounding, float64 enforcement, and run lifecycle
  reconciliation have been integrated with targeted regression coverage.
- The current proof exposed that a successful writer and local probes do not
  establish certification: full-vector `np.equal` and exact scale winners are
  the gate.
- High-level Python SLAVV facade created (`slavv_python/pipeline/slavv_vectorize.py` + stage managers) providing `vectorize_python` (orchestrator equivalent of `vectorize_V200.m`) and thin `get_*_python` convenience wrappers. The exact-parity logic lives in the stage managers and `matlab_get_*` modules. See `PARITY_RANDOM_COMPONENT_SUITE.md` for details.

## Navigation

| Need | Document |
|---|---|
| Current successor brief | [HANDOFF.md](../.agents/HANDOFF.md) |
| Active tasks | [TODO.md](TODO.md) |
| Exact proof evidence | [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md) |
| Phase 1 requirements | [phase-1-exact-route-spec.md](plans/phase-1-exact-route-spec.md) |
| Operator workflow | [PARITY_PRE_GATE.md](reference/workflow/PARITY_PRE_GATE.md) |
