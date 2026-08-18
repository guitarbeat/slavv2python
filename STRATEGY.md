---
name: SLAVV Python
last_updated: 2026-08-17
---

# SLAVV Python Strategy

## Target problem

Vascular imaging labs need vessel centerline graphs from noisy 3D two-photon volumes that match the published SLAVV method. The method of record lives in MATLAB; ports silently change topology because Energy floats and column-major tie-breaking are not interchangeable with “close enough” NumPy.

## Our approach

Certify a MATLAB-faithful exact route against preserved oracles with defined per-stage bars, freeze that baseline, then optimize or accelerate inner loops only behind those bars. Eventual C++ is a native extension of sequential bottlenecks, not a second unverified rewrite.

## Who it's for

**Primary:** Maintainer / developer - They're hiring SLAVV Python to produce MATLAB-equivalent vessel networks from 2P volumes without staying on MATLAB.

**Secondary:** Public pipeline user - They're hiring `slavv run` (paper profile) to extract a `network.json` without running the parity harness. That path is not yet certified at the same bar.

## Key metrics

- **Exact-route certification** - Per-stage `prove-exact` pass/fail on the claim root; live in ONE TRUTH, not this file.
- **Stretch status** - Extra “identical last digits” bar after Phase 1. Live in dest `stretch_status.json` (`blocked_float_path` = Energy still not bit-equal). Close-enough (`allclose`) is not stretch success.
- **Paper-profile certification** - Same sequential bars on the public `paper` profile (phase-1 spec R7). Unmet until a volume/oracle is named and proved.
- **Stage wall-clock** - Frozen-dest Energy→Network timings in `docs/reference/core/phase2-profiling-baseline.json` (can regress).

## Tracks

### Trust / equivalence

Keep Python scientifically equivalent to MATLAB: exact-route stretch (bit-equal Energy, then discrete) and paper-profile certification.

_Why it serves the approach:_ Without a frozen, proved baseline, speed and C++ work cannot be trusted.

### Performance and native speed

Bit-preserving parallelism, profiled stage speedups, optional Fortran unwind only after a Phase 2 ADR, and C++ (`nanobind`/`pybind11`) for sequential inner loops.

_Why it serves the approach:_ The certified route is correct-enough but compute-heavy; acceleration is allowed only when it cannot silently break topology.

### Breadth

Additional real volumes (`neurovasc-db`) and later packaging/UX, after the exact route is trusted.

_Why it serves the approach:_ One canonical volume is the cert surface, not the whole scientific claim.

## Milestones

- **2026-08-17** - Phase 1 exact-route certification closed on full `180709_E`; baseline freeze recorded.

## Not working on

- Reopening Phase 1 or overwriting protected dests (`canonical_full_v18`, `v16`, `crop_M_exact_v3`, `crop_M_stretch_engine_v2`).
- Stretch U5/U6, Fortran unwind, paper-profile writers, or neurovasc-db writers without the documented unlock/ADR.
- Calling allclose, ownership %, or matlab2python “100% parity”; Network rewrite as the default residual fix.
