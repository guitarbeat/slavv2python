# Parity Loop Tuning — Strategy Digest

**Run:** `2026-09-01T06:17:18Z-ce-optimize`  
**Branch:** `optimize/parity-loop-tuning`  
**Phase:** Pre-octave-3 (MATLAB octave 2/6, crop SIGSTOP)

## Spec adjustments (Phase 1.7)

- Added `metric.phase_aware_primary` with three phases: `pre_octave3` (informational), `post_matlab_exit` (optimize, target <1 min), `post_cont` (optimize prove latency).
- Moved `parity_loop_measure.sh` to mutable scope; added `parity_orchestrator.log` to immutable (append-only contract).
- Marked `poll-consolidation` tuning candidate **completed**.
- Added constraints: no legacy watcher reintroduction; tier-2 not expected on 8GB — optimize tier-3 instead.
- Measurement notes: harness reads `parity_orchestrator.log`, not retired dual/octave3 logs.

## Experiments (5/4 max — all kept)

| # | Hypothesis | Outcome | Key effect |
|---|------------|---------|------------|
| 1 | Tier-3 same-tick CONT | **Kept** | Adaptive sleep: 30s while waiting for MATLAB exit, 0s when dead |
| 2 | Orchestrator log → harness | **Kept** | `parity_loop_measure.sh` reads orchestrator log |
| 3 | RESUME_LIVE_STATUS metadata | **Kept** | Status references orchestrator; legacy watchers nulled |
| 4 | Post-CONT auto-prove | **Kept** | Prove script invoked each orchestrator tick (idempotent) |
| 5 | Idle poll 900s | **Kept** | Agent throttle 1800→900s |

## Gates (all green)

- `watcher_alive`: 1 (orchestrator PID 95824)
- `matlab_segfault_recent`: 0
- `duplicate_resume_exact_run`: 0

## Primary metric interpretation

Current phase is **pre_octave3** → `minutes_blocked_before_octave3 = 0.0` (informational).  
Post-MATLAB-exit target (`minutes_to_cont_after_matlab_exit < 1`) becomes measurable when MATLAB PID 27486 exits. Expected improvement: max tier-3 detection latency drops from **300s → 30s** (waiting poll) plus **0s** same-tick CONT on death detection.

## Orchestrator restart

**Required** for live loop (PID 95824) to pick up adaptive sleep and post-cont prove wiring.  
Safe restart when MATLAB/crop are healthy:

```bash
bash workspace/scratch/parity_orchestrator.sh detach
```

`detach` double-forks a new loop; does **not** kill MATLAB (27486) or crop (27480). Old loop PID will exit on next cycle or can coexist briefly — verify with `parity_orchestrator.sh status`.

## Files changed

- `.context/compound-engineering/ce-optimize/parity-loop-tuning/spec.yaml`
- `.context/compound-engineering/ce-optimize/parity-loop-tuning/experiment-log.yaml`
- `workspace/scratch/parity_orchestrator.sh`
- `workspace/scratch/dual_parity_resume_watcher.sh`
- `workspace/scratch/parity_loop_measure.sh`
- `workspace/scratch/goal_idle_mode.json`
- `workspace/scratch/GOAL_AGENT_THROTTLE.json`

## Recommended next steps

1. **Restart orchestrator** via `detach` when convenient (after current tick completes).
2. **Monitor** tier-3 at MATLAB exit: grep `TIER3 matlab_dead_immediate` in `parity_orchestrator.log`; expect CONT within same tick.
3. **Re-measure** post-exit: `bash workspace/scratch/parity_loop_measure.sh` should report `minutes_to_cont_after_matlab_exit`.
4. **Do not** re-enable legacy watchers or tier-2 RAM tuning on 8GB unless tier-3 proves insufficient.
