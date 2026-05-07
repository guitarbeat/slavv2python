# Investigation Archive

[Up: v22 Pointer Corruption Archive](README.md)

This file preserves the April 2026 v22 global watershed investigation as one
narrative archive. It replaces the older quickstart, handover, bug-fix,
blocking-bug, and pointer-investigation notes.

## Symptoms And Scope

The investigation started after a native-first `capture-candidates` run showed
that the v22 global watershed route was still far from MATLAB parity.

Primary measured gap at the time:

| Metric | Count |
| --- | --- |
| MATLAB candidates | 2533 |
| Python candidates | 2120 |
| Matched pairs | 1643 |
| Missing MATLAB pairs | 890 |
| Extra Python pairs | 477 |

The earliest blocking symptoms looked worse than a simple count gap:

- repeated cycle-detection failures during candidate backtracking
- pointer indices far outside the valid LUT size for the active scale
- a working theory that v22 could not even generate valid candidate traces

Characteristic archived error shapes:

```text
ERROR:root:Cycle detected in global watershed backtrack at <location>. Breaking.
ERROR:root:Pointer index 1373 out of range for scale 6 (size 81) at <location>.
Trace history: [(12470094, 328, 11), (12532290, 1373, 6)]
```

## Investigation Timeline

### 1. Blocking-bug triage

The first archive notes treated the v22 route as blocked by pointer and cycle
failures. The immediate task was to figure out whether the LUTs themselves were
wrong or whether runtime pointer ownership was broken.

Important early result:

- `prove-luts` passed, so the LUT construction itself was not the first bug

### 2. Defensive-filtering hypothesis was tested and rejected

A first suspicion was that Python-side defensive bounds filtering was rejecting
otherwise valid MATLAB-style pointers.

That explanation did not hold up. The archived diagnostics showed that the
worst pointers were already invalid before any filtering step could explain the
remaining parity gap.

### 3. Pointer creation and write-path diagnostics

The next phase instrumented the Python watershed path heavily:

- assertions in current-strel construction
- validation before reveal/write
- immediate write-readback checks
- overwrite detection on shared-state maps
- trace-history logging during backtracking

Those checks narrowed the problem substantially:

- pointer indices were valid when created
- pointer indices were valid before reveal
- immediate write-readback did not show corruption
- the mismatch was surfacing later, during trace-back interpretation

### 4. Scale-mismatch diagnosis and pointer-lifecycle fixes

The most productive archived diagnosis was that Python could build a pointer LUT
for one scale while storing a different scale label for later backtracking.
That led to the clipped-scale consistency fix and the broader pointer-lifecycle
cleanup that followed.

### 5. Post-fix reassessment

Once the pointer-lifecycle fixes landed, the investigation moved from
"generation is broken" to "parity is still open even after the obvious pointer
bugs are fixed."

The later archive and maintained docs converged on a new interpretation:

- the reviewed MATLAB and Python watershed constants already matched
- the reviewed size, distance, and direction penalties already matched
- the remaining gap looked more like control-flow drift than scalar mismatch

## Fixes Landed During And Immediately After The Investigation

These fixes were real outputs of the v22 investigation and should remain part
of the exact-route implementation:

- clipped-scale consistency between LUT creation and `size_map` storage
- MATLAB-order linear backtracking for half-edge tracing
- final edge energy and scale sampling on the assembled linear trace
- MATLAB-aligned shared-map dtypes for `pointer_map` and `d_over_r_map`
- later source-backed watershed fixes such as MATLAB-derived scale tolerance and
  MATLAB-style join reset semantics

## Dead Ends And Disproven Hypotheses

The archive is still useful because it shows which theories no longer deserve to
be the first suspect.

### Not the primary current explanation

- a simple defensive-filtering story
- invalid pointer creation at the current-strel stage
- immediate corruption during pointer writes
- a broad MATLAB-vs-Python scalar-parameter mismatch
- a broad size, distance, or direction penalty-formula mismatch

### Useful caution from the archive

Some of the interim docs over-weighted the pointer-corruption framing. That was
reasonable during the first failing runs, but it eventually became too narrow.
The maintained status moved away from that explanation once the parameter and
penalty reviews came back aligned.

## Historical Commands And Logs Worth Preserving

These were the core investigation commands used during the April 2026 funnel:

```powershell
python workspace/scripts/cli/parity_experiment.py preflight-exact `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\20260421_accepted_budget_trial `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\my_current_code_trial

python workspace/scripts/cli/parity_experiment.py prove-luts `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\20260421_accepted_budget_trial `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\my_current_code_trial

python workspace/scripts/cli/parity_experiment.py capture-candidates `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\20260421_accepted_budget_trial `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\my_current_code_trial

python workspace/scripts/cli/parity_experiment.py summarize `
    --run-root D:\slavv_comparisons\experiments\live-parity\runs\my_current_code_trial
```

These commands matter historically because they separate:

- LUT proof
- candidate capture
- downstream reruns
- proof summaries against preserved MATLAB vectors

## Archived Bottom Line

The v22 investigation found real pointer-lifecycle bugs and produced valuable
fixes, but those fixes did not fully explain the remaining parity gap.

The maintained conclusion has moved forward:

- keep the landed pointer-lifecycle fixes
- do not re-open scalar-parameter or penalty-formula mismatch as the first
  theory without new evidence
- treat frontier, join, sentinel, and chooser control flow as the main open
  downstream parity surfaces
