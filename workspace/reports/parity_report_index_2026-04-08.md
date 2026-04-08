# Parity Report Index 2026-04-08

This index organizes the April 8, 2026 parity notes into a compact reading
order.

## Read First

- [Parity Decision Memo 2026-04-08](parity_decision_memo_2026-04-08.md)
  - shortest summary
  - current decision
  - best next implementation step

## Read Second

- [MATLAB/Python Code Audit 2026-04-08](matlab_python_code_audit_2026-04-08.md)
  - Side-by-side MATLAB/Python code audit
  - Cleanup-path mismatch
  - Official-doc checks
  - Best next implementation target

## Read Third

- [Imported-MATLAB Parity Evidence 2026-04-08](parity_investigation_notes_2026-04-08.md)
  - Fresh rerun baseline
  - Candidate coverage
  - Shared-vertex evidence
  - Current artifact-driven diagnosis

## Short Version

- The fresh run still fails parity at `1425` Python edges vs `1379` MATLAB
  edges.
- The candidate pool is still wrong upstream.
- The Python cleanup path is also modeling the wrong MATLAB surface.
- The best next code change is to replace the current Python cleanup path with
  the active MATLAB V200 cleanup chain, then remeasure the remaining gap.

## Files

- `workspace/reports/parity_decision_memo_2026-04-08.md`
- `workspace/reports/parity_investigation_notes_2026-04-08.md`
- `workspace/reports/matlab_python_code_audit_2026-04-08.md`
