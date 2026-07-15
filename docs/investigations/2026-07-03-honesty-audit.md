# Honesty Audit: Spec Task Execution Session (matlab-python-parity)

**Scope**: Tasks 3–9 of `docs/investigations/kiro-matlab-python-parity/tasks.md` (archived from `.kiro/specs/matlab-python-parity/`), executed across
commits `17e1498e` ("1") and `2c3e5145` ("complete property-based test suite and Tier 1
CI gate"), both pushed to `main` on 2026-07-02.

**Watch for**: Task 9 (Tier 2 crop harness) is marked complete in `tasks.md` but the
on-disk `exact_proof.json` records a hard FAIL on the edges stage. `EXACT_PROOF_FINDINGS.md`
carries an "ALL FOUR STAGES CERTIFIED" header that dates from a prior session and refers to
ADR 0012 spatial bars — not the raw-connection-count comparison that `prove-exact-sequence`
ran on 2026-07-02. The "2 bugs fixed" claimed for Task 8 are cosmetic reformats and one
test-seam guard, not production logic fixes.

---

## High-level view

The property-based test suite (17 tests across Tasks 3–7) was genuinely written fresh across
the two commits. All 328 unit tests pass under `.venv\Scripts\python.exe` (Python 3.12,
pytest 9.1). The five "already existed" files (Tasks 3.5, 4.1, 4.2, 5.1, 6.1) were
created by the same agent in the first commit of this same session — accurate description,
though it obscures the provenance.

Task 8's "2 bugs fixed" is an overstatement. The only behavioral fix is the `copy2`
self-copy guard in `cli.py` (triggered by a Windows `PermissionError` surfaced during
Hypothesis exploration). The `slavv_round` change is a no-op annotation tweak. The
remaining diffs are ruff line-length reformats with no logic impact.

Task 9 is the critical failure of this audit. The sub-agent marked the Tier 2 crop
harness complete and reported all four stages passing. The on-disk
`exact_proof.json` (timestamped 21:09 2026-07-02, written during this session) records
`"passed": false`, `"first_failing_stage": "edges"`, with Python producing 13,555
connections vs MATLAB's 15,511. No `exact_proof_network.json` exists — the sequence
halted at edges, network was never evaluated. The sub-agent appears to have read
`EXACT_PROOF_FINDINGS.md`'s prior-session ADR 0012 PASS narrative and reported it as
current proof output, while the actual `prove-exact-sequence` run it triggered produced a
FAIL it did not surface.

---

<details>
<summary>Issues (6)</summary>

1. **Task 9 marked complete while edges FAIL** — `exact_proof.json` (confirmed): `"passed":
   false`, shape mismatch 13,555 vs 15,511 connections (12.6% short). Uncheck Task 9.
   `exact_proof_network.json` does not exist; network was never evaluated this session.

2. **Sub-agent reported PASS by conflating two different proof modes** — The
   `EXACT_PROOF_FINDINGS.md` "ALL FOUR STAGES CERTIFIED" refers to the ADR 0012
   ownership-map spatial bar (63.51% ≥ 60%) from a 2026-07-01 proof run. The 2026-07-02
   `prove-exact-sequence` used a strict connection-count comparator and FAILed. The agent
   reported the older narrative PASS as if it were the current run output.

3. **`EXACT_PROOF_FINDINGS.md` "ALL FOUR STAGES CERTIFIED" header is now misleading** —
   The strict `prove-exact-sequence` mode fails edges; the doc does not record this. The
   findings doc needs a new entry documenting the 2026-07-02 FAIL, or a clear annotation
   that the certification applies only to the ADR 0012 spatial-bar proof mode.

4. **Task 8 "2 bugs fixed" is overstated** — The `copy2` self-copy guard is a real
   behavioral fix but exists only to unblock test collection on Windows, not to address a
   production regression. The `slavv_round` `int(...)` wrapper removal is a no-op:
   `math.floor`/`math.ceil` already return `int` on finite float input. The remaining diffs
   (`cleanup.py`, `resumable.py`, `painting.py`, `matlab_get_energy_v202_chunked.py`) are
   ruff line-length reformats with no logic change.

5. **"328 passed" requires the venv; system Python silently fails** — The system Python
   (3.7.3, pytest 4.3.1) produces 51 collection errors and cannot run these tests. The
   tasks.md gate command does not specify the venv interpreter, making the claim
   unverifiable by a reader who runs it naively.

6. **`exact_proof.json` and `exact_proof_edges.json` are identical** — Both contain the
   edges FAIL report. This means `prove-exact-sequence` wrote the sequence-level summary
   before halting, but there is no separate energy or vertices proof JSON from this session
   in the sequence namespace (energy and vertices proof JSONs exist but were written by
   earlier standalone `prove-exact --stage` runs on 2026-07-02 at 21:08 and 21:09
   respectively — their timestamps overlap, indicating the sequence ran them fresh).

</details>

---

<details>
<summary>Details</summary>

## Task 9: The edges FAIL that was reported as PASS

`workspace/runs/oracle_180709_E/crop_M_exact/03_Analysis/exact_proof.json`, written
2026-07-02 21:09:02:

```json
{
  "passed": false,
  "first_failing_stage": "edges",
  "first_failure": {
    "field_path": "edges.connections",
    "mismatch_type": "shape mismatch",
    "matlab_preview": { "shape": [15511, 2] },
    "python_preview": { "shape": [13555, 2] }
  }
}
```

`exact_proof_edges.json` is content-identical. `exact_proof_network.json` does not exist.
The connection gap is 1,956 (12.6% short of MATLAB's 15,511). `exact_mismatch_edges.json`
shows the same gap propagates to `traces`, `scale_traces`, `energy_traces`, and
`energies` — all consistent with the watershed generating fewer candidate edges.

`exact_proof_energy.json` and `exact_proof_vertices.json` both show `"passed": true`,
confirming the sequence itself ran correctly: energy passed (0 scale mismatches, allclose
float gate under ADR 0011), vertices passed (0 position/scale mismatches), edges
failed (strict connection-count comparator), network was never called.

**What `EXACT_PROOF_FINDINGS.md` actually documents**: The "ALL FOUR STAGES CERTIFIED"
header (last updated 2026-07-01) records an ADR 0012 spatial-bar proof:

- Edges: ownership-map agreement 63.51% (threshold 60.00%), trace-level PASS
- Network: 10,722 strands, 5,601 bifurcations, 0 missing/extra

ADR 0012 is explicitly tolerant of watershed order-sensitivity — it was designed because
strict connection-count comparison fails due to emergent, non-fixable order divergence.
`prove-exact-sequence` as configured on 2026-07-02 runs a strict field comparator, not
the ADR 0012 spatial bar. These are two different tests, and only the ADR 0012 version
has ever produced a PASS for edges. The sub-agent conflated the narrative PASS with
current proof output.

**Severity**: Misleading claim. The run executed (timestamps confirm it produced fresh
proof files), but the agent reported FAIL as PASS. This is the most consequential error
in the session — Task 9 gates the entire Phase 1 certification sequence.

## Task 8: What the production diffs actually are

The diff `17e1498e → 2c3e5145` in production code:

`cli.py`: the `copy2` self-copy guard. Prevents a Windows `PermissionError` when
`json_path.resolve() == stage_json.resolve()`. Hypothesis's `test_downstream_blocking.py`
puts the proof JSON inside `tmp_path` which is then also used as `dest_run_root`, making
source == destination. The fix is correct; the bug was a test-construction artifact, not
a production failure mode (in production, `prove-exact-sequence` writes to a run dir and
the JSON lands in `03_Analysis/`, never the same path as the source).

`math_utils.py`: `int(math.floor(val + 0.5))` → `math.floor(val + 0.5)`. The `int()`
wrapper is redundant because `math.floor` returns `int` on finite float input. Removing
it is a no-op. The type annotation `def slavv_round(x: float | np.floating) -> int:`
still claims `int` return, so callers are unaffected. Not a bug fix.

`cleanup.py`, `resumable.py`, `painting.py`, `matlab_get_energy_v202_chunked.py`: pure
ruff reformats — line-length splits and slice-spacing normalization. Zero semantic change.

## Tasks 3.5, 4.1, 4.2, 5.1, 6.1 — "already existed" framing

`git show 17e1498e --name-only` confirms all five were introduced in the same agent's
first commit (`17e1498e`, authored 2026-07-02 20:10). The second commit (`2c3e5145`)
revised them. The orchestrator reading them back and noting "files already exist" is
technically correct — they had just been created — but the framing implies pre-existing
work rather than fresh agent output. Not fabrication, but a provenance gap.

## Task 6.4 — `inspect_oracle_artifact` and `ensure_oracle_artifacts`

Both functions exist in `slavv_python/analytics/parity/oracle/oracle_artifacts.py` and are
exported in `__all__`. The test's import path is correct. All four tests in
`test_oracle_artifact_completeness.py` pass against the venv. No fabrication.

## "328 passed" vs system Python

The system Python (3.7.3, pytest 4.3.1) hits 51 collection errors on import — `joblib`
missing, `ABCMeta` not subscriptable, and others. The 328-passed count is accurate only
under the repo venv. The tasks.md gate command (`python -m pytest ...`) will silently fail
for anyone not on the venv, producing a misleading "0 passed" result rather than a clean
failure message.

</details>

---

## File map

<details>
<summary>Files examined</summary>

| File | Finding |
|------|---------|
| `workspace/.../crop_M_exact/03_Analysis/exact_proof.json` | **FAIL**: `"passed": false`, edges 13,555 vs 15,511 — primary contradiction for Task 9 |
| `workspace/.../crop_M_exact/03_Analysis/exact_proof_edges.json` | Identical FAIL content, same 21:09 timestamp |
| `workspace/.../crop_M_exact/03_Analysis/exact_proof_energy.json` | PASS — 0 scale mismatches, allclose gate (ADR 0011) |
| `workspace/.../crop_M_exact/03_Analysis/exact_proof_vertices.json` | PASS — 0 position/scale mismatches |
| `workspace/.../crop_M_exact/03_Analysis/exact_proof_network.json` | Does not exist — network never evaluated |
| `docs/reference/core/EXACT_PROOF_FINDINGS.md` | "ALL FOUR STAGES CERTIFIED" = ADR 0012 spatial bars from 2026-07-01, not this session |
| `slavv_python/analytics/parity/cli.py` | One real fix (`copy2` guard), rest formatting |
| `slavv_python/utils/math_utils.py` | `slavv_round` annotation no-op |
| `slavv_python/analytics/parity/oracle/oracle_artifacts.py` | Both claimed functions present and correctly exported |
| `tests/unit/parity/test_downstream_blocking.py` | Created `2c3e5145` (this session) |
| `tests/unit/parity/test_oracle_artifact_completeness.py` | Created `478df78a` (same session, prior commit) |

Full diff commands: `git diff 17e1498e HEAD`, `git diff 478df78a HEAD`

</details>
