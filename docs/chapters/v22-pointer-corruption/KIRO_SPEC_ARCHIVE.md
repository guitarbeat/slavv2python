# Kiro Spec Archive

[Up: v22 Pointer Corruption Archive](README.md)

This file merges the former Kiro bugfix, design, and task docs into one archive.
It preserves the planning language that accompanied the v22 investigation while
making clear which parts stayed true and which parts became stale.

## Original Kiro Framing

The original Kiro spec set framed the remaining v22 gap mainly as a
pointer-corruption problem:

- scale clipping could build a LUT for one scale while storing a different
  scale label
- pointer traces could later be interpreted against the wrong LUT
- once that was fixed, parity was expected to move close to MATLAB

That framing was useful because it drove several real fixes, but it eventually
proved too narrow for the maintained codebase.

## What Stayed True

These parts of the old Kiro framing still hold and are now part of the live
exact-route understanding:

- clipped-scale consistency matters and should stay fixed
- MATLAB-order linear backtracking and final energy or scale sampling matter
- MATLAB-aligned shared-state dtypes matter on parity-sensitive maps
- the public `number_of_edges_per_vertex = 4` surface is not the controlling
  watershed constant on the exact route
- the reviewed MATLAB and Python watershed constants already align
- the reviewed size, distance, and direction penalties already align

## What Became Stale

The old Kiro material no longer reflects the best maintained read in a few key
ways.

### Too narrow

It treated pointer corruption as the main remaining defect even after the live
route had absorbed the clipped-scale and trace-sampling fixes.

### Too optimistic about queue approximations

It did not fully separate literal MATLAB control flow from faster Python
approximations. The maintained parity work now treats frontier ordering, join
cleanup, vertex sentinel handling, and chooser trace order as first-class audit
surfaces.

### Too tolerant of investigation-era logic in production parity code

The archive-era task list still allowed diagnostic logging and defensive
pointer filtering to live inside the canonical exact path for too long. The
maintained docs now treat those as temporary aids, not part of the claim
surface.

## Final Archived Task List

### Completed and still valid

- keep clipped-scale consistency between LUT creation and `size_map` storage
- keep MATLAB-order linear backtracking for half-edge tracing
- keep final edge energy and scale sampling on the assembled linear trace
- keep MATLAB-aligned dtypes for parity-sensitive shared-state maps
- record the parameter-parity review in maintained docs
- record the penalty-parity review in maintained docs

### Still relevant, but now tracked in maintained docs instead of Kiro files

- replace remaining frontier approximations with MATLAB vector semantics where
  exact-route claims require it
- preserve MATLAB-style join reset semantics
- preserve the vertex `-Inf` sentinel lifecycle
- remove or isolate diagnostic-era production logic from the canonical exact
  path
- re-run native-first `capture-candidates` and `prove-exact` after each real
  parity-bearing change

### Retired or re-scoped

- treat scalar parameter mismatch as the first explanation for the remaining gap
- keep separate quickstart, handover, bug-fix, and blocking-bug docs for the
  same v22 story
- keep the archived Kiro plan as a maintained status surface instead of moving
  the live status into `EXACT_PROOF_FINDINGS.md`

## Archive Note

The raw Kiro config remains preserved at `kiro-spec/.config.kiro`. The live
status and roadmap now belong in the maintained reference docs rather than in
separate Kiro planning files.
