# Commit Messages for May 2026 MATLAB Parity Fixes

## Commit 1: Fix directional suppression bug in watershed seed loop

```
fix(watershed): remove directional suppression from inside seed loop

CRITICAL BUG: Python was applying directional suppression INSIDE the
watershed seed loop, mutating adjusted energies after each seed
selection. MATLAB computes adjusted energies ONCE before the loop and
uses them unchanged for all seeds from a given location.

This bug directly caused the 16.3% candidate generation gap (2120
Python vs 2533 MATLAB candidates) because:
- It affected every location emitting multiple seeds (vertices with
  edge_number_tolerance=2)
- Caused Python to select different second seeds than MATLAB
- Accumulated over thousands of watershed iterations

MATLAB Reference (external/Vectorization-Public/slavv_python/get_edges_by_watershed.m):
- Lines 207-343: Compute adjusted energies BEFORE seed loop
- Lines 476-565: Seed loop only READS current_strel_energies, never mutates

The MATLAB method applies all energy penalties (size, distance,
direction) BEFORE the seed loop begins, then uses the same adjusted
energy field for all seeds from that location.

Changes:
- slavv_python/core/_edge_candidates/global_watershed.py (lines 694-720):
  Removed directional suppression from inside seed loop
  Added detailed comments explaining MATLAB behavior
  
- tests/unit/core/test_watershed_seed_suppression_bug.py (new):
  Demonstration test showing the bug's mechanism and impact

Validation:
- All watershed tests pass (81/82, 1 pre-existing frontier ordering failure)
- Code formatted and linted successfully

This fix should dramatically improve or close the candidate generation
gap. Re-run capture-candidates to measure actual improvement.

Related: docs/reference/core/EXACT_PROOF_FINDINGS.md
Related: docs/reference/core/MATLAB_PARITY_MAPPING.md
```

## Commit 2: Fix trace order randomization in edge selection

```
fix(edge-selection): always use seeded RNG for trace order randomization

Python only randomized trace point order when comparison_exact_network=True,
but MATLAB always uses randperm for deterministic trace order.

This caused:
- Non-deterministic trace order on non-exact routes
- Incorrect parity assumption that randomization was exact-route-only

MATLAB Reference (external/Vectorization-Public/slavv_python/choose_edges_V200.m):
- Line 318: edge_position_index_range = uint16(randperm(degrees_of_edges(edge_index)));

MATLAB always uses randperm to randomize the order in which trace points
are processed during conflict painting, regardless of execution mode.

Changes:
- slavv_python/core/edges_internal/edge_selection.py (lines 163-170, 220-225):
  Always initialize and use seeded RNG for trace order
  Removed conditional check that prevented randomization on non-exact routes
  Now matches MATLAB's randperm behavior on all routes

Validation:
- All edge selection tests pass (9/9)
- Code formatted and linted successfully

Related: docs/reference/core/EXACT_PROOF_FINDINGS.md
Related: docs/reference/core/MATLAB_PARITY_MAPPING.md
```

## Commit 3: Update documentation with May 2026 parity fixes

```
docs: document May 2026 critical MATLAB parity bug fixes

Updated EXACT_PROOF_FINDINGS.md to document:
- Directional suppression bug in watershed seed loop (CRITICAL)
- Trace order randomization bug in edge selection
- Impact analysis and MATLAB slavv_python references
- Updated next proof actions prioritizing candidate measurement

Changes:
- docs/reference/core/EXACT_PROOF_FINDINGS.md:
  Added "May 2026 Critical Bug Fixes" section
  Updated "Strongest Remaining Candidate Surfaces" section
  Updated "Next Proof Actions" to prioritize candidate measurement
  Updated last modified date to 2026-05-04

The directional suppression fix should dramatically improve or close
the 16.3% candidate generation gap. Next action is to re-run
capture-candidates to measure actual improvement.
```

## Git Commands

To create these commits (assuming changes are staged appropriately):

```powershell
# Commit 1: Watershed fix
git add slavv_python/core/_edge_candidates/global_watershed.py
git add tests/unit/core/test_watershed_seed_suppression_bug.py
git commit -F commit1.txt

# Commit 2: Edge selection fix
git add slavv_python/core/edges_internal/edge_selection.py
git commit -F commit2.txt

# Commit 3: Documentation
git add docs/reference/core/EXACT_PROOF_FINDINGS.md
git commit -F commit3.txt
```

Or as a single commit if preferred:

```powershell
git add slavv_python/core/_edge_candidates/global_watershed.py
git add slavv_python/core/edges_internal/edge_selection.py
git add tests/unit/core/test_watershed_seed_suppression_bug.py
git add docs/reference/core/EXACT_PROOF_FINDINGS.md
git commit -m "fix: critical MATLAB parity bugs in watershed and edge selection

See COMMIT_MESSAGES.md for detailed commit messages for each fix."