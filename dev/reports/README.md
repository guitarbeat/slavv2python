# Reports Catalog

## What this file is for

`dev/reports/` now holds a small canonical report set for the current
parity and release-operations story. The goal is fast orientation: a reader
should be able to tell what is broken, what is already solved, and what to do
next without opening a long chain of overlapping notes.

## Read this when

- you need the current parity diagnosis and next implementation target
- you need the strongest supporting technical evidence
- you need the repeatability baseline for MATLAB and Python runs
- you need the operator runbook for April 13 release/parity execution issues

## Executive Summary

- Reports are now grouped by handling status under `handled/` and
  `unhandled/`.
- The parity story is split cleanly into a fast decision memo, a technical
  audit appendix, and a repeatability baseline.
- The April 13 release/rerun incident now lives in a single operator runbook
  instead of separate attempt and incident notes.
- Raw machine-heavy outputs should live under `tooling/`, not in new top-level
  narrative reports unless they add genuinely new evidence.

## Current Status

| Report | Status | Notes |
| --- | --- | --- |
| [handled/parity_decision_memo_2026-04-08.md](handled/parity_decision_memo_2026-04-08.md) | Handled | Canonical orientation memo for the April 7 to April 8 diagnosis set; decision scope is complete. |
| [handled/matlab_python_code_audit_2026-04-08.md](handled/matlab_python_code_audit_2026-04-08.md) | Handled | Technical appendix is complete; cleanup-chain alignment recommendation has been implemented. |
| [handled/python_nondeterminism_investigation_2026-03-28.md](handled/python_nondeterminism_investigation_2026-03-28.md) | Handled | Repeatability investigation and deterministic fix baseline. |
| [handled/file_lock_contention_analysis_2026-04-13.md](handled/file_lock_contention_analysis_2026-04-13.md) | Handled | April 13 operational incident analysis and recovery runbook. |
| [handled/release_verification_2026-04-14.md](handled/release_verification_2026-04-14.md) | Handled | Release verification closure and performance snapshot. |

- Unhandled set: none currently cataloged
- Handled set: `handled/parity_decision_memo_2026-04-08.md`,
  `handled/python_nondeterminism_investigation_2026-03-28.md`,
  `handled/matlab_python_code_audit_2026-04-08.md`,
  `handled/file_lock_contention_analysis_2026-04-13.md`, and
  `handled/release_verification_2026-04-14.md`
- Retired set: older overlapping parity notes, consistency checkpoints, and
  the separate April 13 release attempt log have been folded into the current
  set and removed from top-level storage.

## Unhandled reports

No unhandled reports are currently cataloged.

## Handled reports

1. [handled/parity_decision_memo_2026-04-08.md](handled/parity_decision_memo_2026-04-08.md)
   - Canonical orientation memo for the April 7 to April 8 diagnosis set.
   - Use this when you want the shortest statement of what was solved, what was
     open at that time, and which implementation direction was chosen.
2. [handled/python_nondeterminism_investigation_2026-03-28.md](handled/python_nondeterminism_investigation_2026-03-28.md)
   - Canonical repeatability baseline for March 28 to March 30.
   - Use this when you need the pre-fix instability story, the deterministic
     padding fix result, or the MATLAB/Python standalone consistency baseline.
3. [handled/file_lock_contention_analysis_2026-04-13.md](handled/file_lock_contention_analysis_2026-04-13.md)
   - Canonical April 13 release-operations incident and runbook.
   - Use this when you need to classify rerun failures, recover a completed run
     safely, or avoid Windows file-lock contention.
4. [handled/release_verification_2026-04-14.md](handled/release_verification_2026-04-14.md)
   - Canonical release verification closure note for the April 14 audit.
   - Use this when you need final checklist completion and performance snapshot
     references.
5. [handled/matlab_python_code_audit_2026-04-08.md](handled/matlab_python_code_audit_2026-04-08.md)
   - Canonical technical appendix for the April 8 parity diagnosis.
   - Use this when you need the strongest artifact-backed edge cleanup evidence
     and audit rationale behind the parity decision memo.

## Read order

1. Read the handled parity decision memo for the diagnosis baseline and
  implementation-direction rationale.
2. Open the handled code audit if you need the engineering detail behind that
  diagnosis and the evidence trail for the completed cleanup-chain alignment.
3. Use the handled nondeterminism investigation to separate solved
  repeatability work from the still-open semantic parity gap.
4. Use the handled file-lock report for release execution and rerun guidance.
5. Use the handled release verification audit for final checklist closure and
  performance context.

## Retired report map

Several older top-level notes were intentionally folded into the canonical set
above:

- the older April 8 parity index, evidence note, and stage-isolated network
  probe now live in the parity decision memo and the code audit
- the older March 28 to March 30 standalone consistency and post-fix parity
  notes now live in the nondeterminism investigation baseline
- the older April 13 release attempt log now lives in the file-lock incident
  runbook

Their unique conclusions were migrated into the surviving documents before the
redundant files were removed.

## Operator guidance

- Treat `tooling/` as raw supporting output, not the starting point for humans.
- Put future machine-heavy snapshots under `tooling/` when possible.
- Add a new dated top-level report only when it introduces genuinely new
  evidence or a new operational incident that does not fit one of the current
  canonical documents.

