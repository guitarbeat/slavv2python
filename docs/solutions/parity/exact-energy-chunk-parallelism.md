---
title: Exact-Route Energy Chunk Parallelism (n_jobs)
module: pipeline/energy
tags: [performance, parallelism, exact-route, energy, n_jobs, certification]
problem_type: performance
resolution_type: configuration
---

# Exact-Route Energy Chunk Parallelism (n_jobs)

## Problem
The full-volume (512x512x64) exact-route Energy stage ran for an estimated
**~37 hours** on octave 1 alone (~54 h for the full energy→network sequence).
At ~44 s/chunk over octave 1's 3,431 chunks, a Tier-3 certification run was a
multi-day commitment, making iteration impractical.

## Evidence
- `run_snapshot.json` reported energy `progress ~0.13` after ~5.3 h, ETA ~37 h.
- The energy worker process used ~84% of a *single* core (CPU-time ≈ wall-time),
  i.e. it was effectively single-threaded.
- `validated_params.json` showed `"n_jobs": 1`; `_prepare_energy_config`
  (`slavv_python/pipeline/energy/config.py`) defaults `n_jobs` to 1.

## Root Cause
`compute_exact_parity_energy_chunked`
(`slavv_python/pipeline/energy/matlab_get_energy_v202_chunked.py`) already
supports threaded chunk parallelism via
`joblib.Parallel(n_jobs=n_jobs, prefer="threads")`, but the run was launched
with `n_jobs=1`, so every chunk was computed serially. The heavy per-chunk work
is FFT/iFFT (numpy/scipy), which **releases the GIL**, so threads yield real
speedup.

## Why It Stays Bit-Exact
- `joblib.Parallel` returns results **in task order**, and the min-combine merge
  (`energy = where(chunk < master, chunk, master)`) is applied serially in that
  same `c_idx` order — identical to the serial loop.
- `_process_chunk` is a pure function of its inputs (no shared mutable state,
  no RNG in the energy stage), so concurrent execution cannot change values.

## Solution
Pass `--n-jobs <N>` to `slavv parity resume-exact-run` / `launch-exact-run`
(wired through to `resume_exact_run(n_jobs=...)`, which overrides
`params["n_jobs"]`). For the canonical machine (8 logical cores, 16 GB RAM),
`--n-jobs 6` is a good balance (~2.5 GB peak, leaves 2 cores).

```powershell
slavv parity resume-exact-run `
  --dest-run-root workspace\runs\oracle_180709_E\canonical_full_v2 `
  --oracle-root   workspace\oracles\180709_E_full_v2 `
  --force-rerun-from energy --stop-after network --skip-preflight --force `
  --n-jobs 6
```

## Verification
A/B on a small multi-chunk volume built from the canonical params
(`_prepare_energy_config`), comparing `n_jobs=1` vs `n_jobs=4`:
- Energy field and scale-index arrays were **byte-identical** (matching SHA-256).
- ~2.38x faster on 8 chunks.

On the full volume's 3,431-chunk octave 1, observed steady-state throughput was
**~7.8 s/chunk at `n_jobs=6`** (vs ~44 s/chunk serial) — a **~5.5x per-chunk
speedup**, cutting energy from ~37 h to ~9–10 h and the full sequence to
~14–16 h.

## Caveats
- With `n_jobs>1`, the run-dir progress (`02_Energy/resume_state.json` units,
  `run_snapshot.json` progress) is the **merge cursor**, which **lags the parallel
  compute** — joblib computes all chunks concurrently, then the progress callback
  fires during the serial merge. So `resume_state` can read ~600 while the joblib
  log shows ~2800 *computed*. For live energy rate/ETA, use the joblib
  `Done N tasks | elapsed` lines (`scripts/parity_run_throughput.py --log`); for a
  liveness/verdict check, use `scripts/check_parity_run.py --run-dir` (it reads the
  heartbeat **age**, not the lagging cursor). Do **not** derive an ETA from the
  run-dir progress — it lags and produces nonsense.
- The energy stage is **not mid-stage resumable** (`resumable: false`), so
  changing `n_jobs` requires `--force-rerun-from energy` (restart from scratch).
- Peak memory scales ~linearly with `n_jobs` (each thread holds a chunk's
  arrays). Keep `n_jobs * per-chunk-peak` within RAM.

## Follow-Up
- Consider raising the default `n_jobs` for the exact route, or auto-sizing it
  from `os.cpu_count()` with a memory guard.
- The per-scale `gc.collect()` in `_process_chunk` serializes threads under the
  GIL; removing/reducing it is numerically bit-safe and could improve scaling
  further (raises peak memory) — see TODO.
