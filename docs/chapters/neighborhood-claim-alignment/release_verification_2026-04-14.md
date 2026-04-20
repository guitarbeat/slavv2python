# Release Verification Audit (2026-04-14)

This note absorbs the remaining release-verification items that were being
tracked in the temporary repo-root `todo.md`.

## Scope

- Canonical comparison input: `data/slavv_test_volume.tif`
- Historical canonical live run root for this audit:
  `C:\slavv_comparisons\20260413_release_verify\live_canonical_20260413`
- MATLAB batch:
  `01_Input\matlab_results\batch_260413-144432`

Current note:

- This report is a closed April 14 release audit. The active imported-MATLAB
  evidence roots now live under
  `slavv_comparisons/experiments/live-parity/runs/20260418_claim_ordering_trial`
  and
  `slavv_comparisons/experiments/live-parity/runs/20260418_network_gate_trial`.

## Release Checklist Outcome

- Fresh live comparison on a high-free-space local output root: complete
- Final live comparison audit on canonical data: complete
- Performance snapshot for native and parity paths: complete
- Final parity findings summary for the release audit: complete

## Handling Classification

- Status: Handled
- Why: Release-verification documentation and checklist closure for this audit
  are complete.
- Remaining parity algorithm work is tracked in
  the active parity chapter and is not a blocker for this report's closed
  scope.

## Operational Verification

The canonical run root satisfies the operational checks that were still open:

- Preflight metadata exists at `99_Metadata/output_preflight.json` and passed
  with `15.2 GB` free against a `5.0 GB` requirement.
- Resume transparency is visible in both `99_Metadata/matlab_status.json` and
  `99_Metadata/run_manifest.md`.
- The final resume state is `complete-noop`, with the prediction that
  `batch_260413-144432` is already complete and a rerun should be a no-op
  unless inputs change.
- Parity diagnostics were generated, including `comparison_report.json`,
  `summary.txt`, and
  `02_Output\python_results\stages\edges\candidate_audit.json`.
- Final edge and strand status is easy to interpret from `summary.txt`:
  vertices passed, while edges and strands failed.

## Canonical Comparison Findings

The canonical run is useful as a release-audit artifact because it confirms the
workflow surfaces are working on the real input, even though exact parity
remains open.

Result summary from `summary.txt` / `comparison_report.json`:

- Vertices: MATLAB `2,577`, Python `2,577`, difference `0`
- Edges: MATLAB `2,533`, Python `46`, difference `-2,487`
- Strands: MATLAB `1,120`, Python `45`, difference `-1,075`
- Parity gate: vertices `PASS`, edges `FAIL`, strands `FAIL`
- First missing candidate endpoint pair: `[0, 23]`
- Candidate source breakdown: fallback-only dominated
  (`6,145` traced candidates, `46` chosen edges, `0` frontier/watershed/geodesic chosen)

Interpretation:

- The release workflow itself is now inspectable and auditable.
- Exact edge/strand parity is still not release-ready as an algorithmic claim.
- The canonical data continues to point at candidate-generation behavior rather
  than a hidden orchestration failure.

## Performance Snapshot

### MATLAB native path

Source:
`01_Input\matlab_results\batch_260413-144432\timings.json`

- Total: `1945.2s` (`32.42m`)
- Energy: `1867.9s`
- Vertices: `18.3s`
- Edges: `57.4s`
- Network: `1.4s`

### Python parity path

Source:
`99_Metadata/run_snapshot.json` plus stage manifests under
`02_Output\python_results\stages\`

- Python pipeline wall time: `2928s` (`48.80m`)
- Energy plus vertex completion window: `2774s`
- Edge stage completion window: `153s`
- Network completion window: `<1s`

### Readout

- On this canonical release audit, the Python parity path took about `1.51x`
  the MATLAB batch wall time.
- Energy remains the dominant cost center on both sides.
- The performance snapshot is now preserved in-repo instead of being stranded
  only in the external run root.

## Bottom Line

The outstanding release-verification tasks from the temporary checklist are now
closed as documentation work:

- the canonical live run evidence exists and is cited
- the staged metadata and rerun semantics are verified
- the performance snapshot is captured
- the final findings summary is preserved in a maintained report

What remains open is algorithmic parity work, not release-audit bookkeeping.
