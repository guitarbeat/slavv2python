---
title: Pause-Safe MATLAB Batches on macOS
module: scripts/parity
tags: [parity, matlab, macos, launchd, run-monitoring, checkpoints]
problem_type: workflow
resolution_type: runbook
---

# Pause-Safe MATLAB Batches on macOS

## In short

Long MATLAB jobs on macOS must not inherit the Codex/ChatGPT application
lifetime. Install the local LaunchAgent with `macos_matlab_job.py`; let launchd
own the supervisor and LaunchServices own MATLAB. Treat MATLAB's nonce-matched
result file—not a shell exit code—as completion.

## Problem

The R2024b full-volume MATLAB writer was originally a child of the desktop app.
When the Dock force-quit ChatGPT, macOS enumerated that application coalition
and terminated MATLAB plus its process worker and `caffeinate`. The scratch
state remained `running`, so neither the saved PID nor the JSON status reflected
reality.

A direct LaunchAgent using `/usr/bin/python3` also could not traverse LoveSSD:
macOS returned `Operation not permitted`. Storing operational state beside the
external experiment therefore made the independent supervisor fail before it
could launch MATLAB.

## Root cause

There were two independent ownership boundaries:

1. A process launched from Codex/ChatGPT belonged to the app's process
   coalition and could be terminated with the app.
2. A background LaunchAgent's system Python did not inherit the interactive
   application's external-volume permission.

`caffeinate` prevents idle sleep; it does not change either boundary. Likewise,
`open -W` only reports the lifecycle of the opened application and cannot prove
that the MATLAB driver completed successfully.

## Solution

[`scripts/parity/macos_matlab_job.py`](../../../scripts/parity/macos_matlab_job.py)
installs a stdlib-only supervisor below:

```text
~/Library/Application Support/slavv2python/matlab-jobs/<job-id>/
├── config.json
├── macos_matlab_job.py
├── matlab_driver_wrapper.m
└── runtime/
    ├── state.json
    ├── supervisor.log
    ├── matlab.log
    ├── matlab.pid
    ├── attempt_nonce.txt
    ├── matlab_result.json
    └── PAUSE
```

The corresponding plist lives in `~/Library/LaunchAgents/`. launchd starts the
supervisor with PPID 1. The supervisor uses `caffeinate` and `open -W -n -a` so
the MATLAB application is created by LaunchServices with PPID 1 and its own
application identity.

The generated MATLAB wrapper:

- records the actual MATLAB PID locally;
- reads the supervisor's unique attempt nonce;
- executes the legacy driver in the base workspace, so a driver-level `clear`
  cannot erase wrapper cleanup state;
- atomically writes `matlab_result.json` with the same nonce;
- records MATLAB exceptions as a failed result before rethrowing.

The supervisor never scans LoveSSD during its heartbeat. The interactive
operator performs TIFF hash validation before install/resume and performs
best-effort external progress inspection during `status`.

## Commands

Run from the repository root:

```bash
# Default: full 180709_E R2024b reproduction. Installs but does not start.
uv run python scripts/parity/macos_matlab_job.py install

uv run python scripts/parity/macos_matlab_job.py start
uv run python scripts/parity/macos_matlab_job.py status
uv run python scripts/parity/macos_matlab_job.py pause
uv run python scripts/parity/macos_matlab_job.py resume
```

`pause` writes the sentinel and asks the live supervisor to terminate the exact
identity-matched MATLAB PID. If the supervisor died but MATLAB survived,
`pause` can rescue the orphan using the unique installed command path. It does
not delete MATLAB pipeline checkpoints. `resume` revalidates the source hash,
clears the sentinel, and refuses a duplicate live MATLAB writer.

Do not edit or delete runtime files while the job is alive. Do not use
`uninstall` until the job is paused or terminal.

## Verification

The implementation was verified on 2026-09-06 with:

- a short launchd → LaunchServices → MATLAB completion probe;
- a pause request while MATLAB was still starting;
- confirmation that MATLAB, `open`, `caffeinate`, and the supervisor were all
  gone before state became `paused`;
- resume in the same job root with a new nonce;
- a MATLAB-written completion marker matching that nonce;
- 10 focused supervisor tests;
- repository suite: 1,066 passed and 33 skipped;
- documentation integrity: zero errors and zero warnings.

## Evidence and claim boundary

The active full run is an **optional R2024b cross-version reproduction**. Its
writer completing means only that new MATLAB artifacts exist. It becomes a
parity result only after the artifacts are validated, promoted to a fresh
R2024b oracle, and compared against a fresh Python run with evaluated Energy,
Vertices, Edges, and Network proofs.

Phase 1 remains CLOSED on `canonical_full_v18` against
`workspace/oracles/180709_E_full_v2`. Never overwrite those protected roots.

## Limitations

- `caffeinate` cannot keep a closed-lid laptop running and cannot survive power
  loss.
- launchd may cap its displayed exit timeout; durable state is written before
  and after process transitions, but an abrupt power loss can still interrupt
  the current MATLAB operation.
- MATLAB's legacy vectorizer determines checkpoint granularity. Pause is
  process-safe, not an arbitrary instruction-level snapshot.
