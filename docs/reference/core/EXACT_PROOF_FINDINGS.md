# Exact Proof Findings

[Up: Reference Docs](../README.md)

**Last Updated**: 2026-05-04

This is the maintained current-status owner for the native-first exact route.
Use it for live proof status, current v22 watershed readouts, the first failing
field, and the measured effect of parity-bearing fixes.

This file is intentionally developer-facing. It does not define the acceptance
gate for the public `paper` CLI/app workflow.

Use the other core docs for different jobs:

- `MATLAB_METHOD_IMPLEMENTATION_PLAN.md` defines claim boundaries and the
  remaining roadmap.
- `MATLAB_PARITY_MAPPING.md` maps MATLAB functions to the live Python tree and
  records confirmed structural deviations.
- [v22 Pointer Corruption Archive](../../chapters/v22-pointer-corruption/README.md)
  preserves the April 2026 investigation trail and archived Kiro planning.

## Scope

- The canonical exact route is `comparison_exact_network=True` with
  `python_native_hessian` as the canonical exact-compatible energy provenance.
- Preserved MATLAB vectors remain the oracle artifacts for `prove-exact`.
- Maintained parity storage now separates preserved MATLAB truth under
  `oracles/` from disposable reruns under `runs/`.
- `100%` means artifact-level equality against preserved MATLAB vectors, not
  count-level similarity.

## Current Status

| Component | Current state | Proof state | Main blocker |
| --- | --- | --- | --- |
| Native energy | Complete | Canonical exact-compatible source | Keep MATLAB-oracle fixture coverage green |
| Vertices | Runnable on the native-first exact route | Proof pending downstream | Awaiting edge proof |
| Edges | Active parity work | Not exact | Candidate-generation and chooser control flow |
| Network | Source-aligned | Proof pending | Upstream edge parity |

## Current v22 Read

The strongest current interpretation is:

- exact-route proof runs must first pass a saved-params fairness audit before
  candidate or chooser counts are treated as trustworthy parity evidence
- the pointer-lifecycle fixes were real and should stay
- the reviewed MATLAB and Python watershed constants are already aligned
- the reviewed size, distance, and direction penalties are already aligned
- the remaining candidate gap looks more like a frontier, join, or chooser
  control-flow problem than a scalar-parameter problem

### Exact Params Fairness Gate

The maintained exact route now rejects source runs whose saved
`validated_params.json` still carries Python-only parity controls or omits the
required MATLAB-shaped exact settings.

The fairness surface must include both serialized MATLAB settings and released
MATLAB source constants that are not written into `settings/*.mat`. The current
maintained exact bootstrap now records at least these source-level edge
constants explicitly:

- `step_size_per_origin_radius = 1`
- `max_edge_energy = 0`
- `edge_number_tolerance = 2`
- `distance_tolerance_per_origin_radius = 3`
- `energy_tolerance = 1`
- `radius_tolerance = 0.5`
- `direction_tolerance = 1`

Exact-route experiments should now also persist:

- `01_Params/shared_params.json`
- `01_Params/python_derived_params.json`
- `01_Params/param_diff.json`

Current live read on the preserved run root
`20260421_accepted_budget_trial`:

- saved params still include Python-only `parity_*` controls
- `energy_projection_mode` is not explicitly recorded as `matlab`

Until that params surface is cleaned up, the run is not a fully fair
start-from-the-same-settings exact baseline even if later proof artifacts are
available.

## Native Energy

The maintained `hessian` path is now the canonical exact-compatible source for
energy generation.

Maintained native-energy coverage includes:

- projected `energy`
- `scale_indices`
- `energy_4d`
- per-scale Laplacian intermediates
- per-scale valid-mask behavior
- direct versus resumable alignment

This removes runtime dependence on imported MATLAB energy artifacts for the
canonical exact route.

## Vertices

Vertex extraction is source-aligned and downstream-ready on the native-first
exact route. No current evidence suggests that vertices are the first failing
surface.

## Edges: v22 Global Watershed

### Latest Maintained Candidate Snapshot

The last maintained v22 `capture-candidates` read remains:

| Metric | Count | vs MATLAB |
| --- | --- | --- |
| MATLAB candidates | 2533 | 100% oracle |
| Python candidates | 2120 | 83.7% |
| Matched pairs | 1643 | 64.9% match |
| Missing pairs | 890 | 35.1% gap |
| Extra pairs | 477 | 22.5% over |

**Note**: These counts are from before the May 2026 critical bug fixes. Re-run
`capture-candidates` to measure the actual improvement from the directional
suppression and trace order fixes.

### May 2026 Critical Bug Fixes

Two critical MATLAB parity bugs were identified and fixed on 2026-05-04:

#### 1. Directional Suppression in Seed Loop (CRITICAL)

**Bug**: Python was applying directional suppression INSIDE the watershed seed
loop, mutating adjusted energies after each seed selection. MATLAB computes
adjusted energies ONCE before the loop and uses them unchanged for all seeds.

**Location**: `source/core/_edge_candidates/global_watershed.py` lines 714-720

**MATLAB Reference**: `external/Vectorization-Public/source/get_edges_by_watershed.m`
- Lines 207-343: Compute adjusted energies BEFORE seed loop
- Lines 476-565: Seed loop that only READS `current_strel_energies`, never mutates

**Impact**: This bug directly caused the 16.3% candidate generation gap because:
- It affected every location emitting multiple seeds (vertices with `edge_number_tolerance=2`)
- Caused Python to select different second seeds than MATLAB
- Accumulated over thousands of watershed iterations

**Fix**: Removed directional suppression from inside seed loop. MATLAB applies
all energy penalties (size, distance, direction) BEFORE the seed loop begins,
then uses the same adjusted energy field for all seeds from that location.

**Validation**: All watershed tests pass (81/82, 1 pre-existing frontier ordering failure)

#### 2. Trace Order Randomization

**Bug**: Python only randomized trace point order when `comparison_exact_network=True`,
but MATLAB always uses `randperm` for deterministic trace order.

**Location**: `source/core/edges_internal/edge_selection.py` lines 221-224

**MATLAB Reference**: `external/Vectorization-Public/source/choose_edges_V200.m` line 318
```matlab
edge_position_index_range = uint16(randperm(degrees_of_edges(edge_index)));
```

**Impact**: Non-deterministic trace order on non-exact routes; incorrect parity
assumption that randomization was exact-route-only.

**Fix**: Always initialize and use seeded RNG for trace order, removing the
conditional check. Now matches MATLAB's `randperm` behavior on all routes.

**Validation**: All edge selection tests pass (9/9)

### Landed Fixes That Should Stay

The current exact-route watershed path has already absorbed these meaningful
fixes:

- clipped-scale consistency between LUT creation and `size_map` storage
- MATLAB-style join-time reset behavior for `available_locations`
- MATLAB-aligned shared-state dtypes for `pointer_map` and `d_over_r_map`
- direct linear-offset backtracking for half-edge tracing
- final energy and scale sampling directly from the assembled MATLAB-order
  linear trace
- MATLAB-derived scale-tolerance calculation from the first two vessel radii

### What The Latest Review No Longer Supports

The latest review does not support treating any of these as the primary current
explanation:

- a simple MATLAB-vs-Python scalar-parameter mismatch story
- a size, distance, or direction penalty-formula mismatch story
- pointer-generation corruption at creation time
- immediate write/read corruption of `pointer_map_flat`

### Red Herrings To Avoid

These have now wasted enough time that they should be treated as explicit
anti-patterns in future edge investigations:

1. Do not use the vertex fields embedded inside the preserved raw `edges*.mat`
   file as the upstream watershed input surface.
   Those embedded vertices reflect the downstream post-`add_vertices_to_edges`
   surface and can include vertices that are not present in the standalone
   curated vertex artifact.
2. Do not treat candidate coverage against the full final `edges.connections`
   surface as a pure upstream watershed proof.
   The preserved final edges artifact includes downstream bridge and added
   vertex effects, so fail-fast candidate counts should be interpreted with
   that limitation in mind.
3. Do not assume the preserved 2019 oracle artifact package and the later
   public MATLAB source are the same code vintage in every local control-flow
   detail.
   Use released MATLAB source as the canonical implementation reference, but
   record and investigate any artifact-vs-source contradiction instead of
   silently collapsing the two.

### Strongest Remaining Candidate Surfaces

After the May 2026 fixes, the remaining candidate surfaces to investigate are:

1. frontier ordering and insertion semantics (1 test failure suggests work remains)
2. join cleanup semantics
3. vertex `-Inf` sentinel lifecycle behavior
4. any remaining chooser/control-flow deviations downstream of candidate emission

The directional suppression fix should dramatically improve or close the candidate
generation gap. Re-run parity experiments to measure actual improvement.

## Cleanup And Network

The cleanup chain is structurally aligned after the April 2026 audit:

- degree cleanup removal order matches MATLAB
- orphan cleanup terminal union matches MATLAB
- cycle cleanup removes the worst edge per component and prunes vertices in the
  same overall way

That means downstream proof is still blocked primarily by unresolved upstream
edge parity rather than known cleanup-specific bugs.

Network and strand assembly remain source-aligned but proof-pending until edge
parity closes.

## Historical Imported-MATLAB Replay Notes

These measurements came from the older imported-MATLAB replay track before the
native-first exact route became canonical. They are kept here only as historical
context.

### Historical First Exact Failure

- stage: `edges`
- field: `connections`
- MATLAB shape: `2533 x 2`
- Python shape: `1654 x 2`

### Historical Candidate Gap

Before the v22 route:

- raw Python candidates: `2364`
- intersection with MATLAB endpoint pairs: `2054`
- missing MATLAB pairs: `479`
- extra Python pairs: `310`

After the old chosen-edge path:

- final Python chosen edges: `1654`
- final chosen-edge intersection: `1553`
- final missing MATLAB pairs: `980`
- final extra Python pairs: `101`

### Historical Cleanup-Gate Fix

Removing the stale nonnegative-energy cleanup gate improved the historical
chosen-edge path from `1553` matched MATLAB pairs to `1886`, but it still did
not close parity.

## Next Proof Actions

1. **IMMEDIATE**: Re-run native-first `capture-candidates` to measure the actual
   improvement from the May 2026 directional suppression fix. This should
   dramatically improve or close the 16.3% candidate generation gap.
2. Re-run `prove-exact --stage edges` and record the first failing field.
3. If candidate generation now matches, investigate the 1 failing frontier
   ordering test as the next parity surface.
4. Clean the exact source-run params surface so preflight passes the fairness
   audit with no Python-only `parity_*` controls.
5. Keep `MATLAB_PARITY_MAPPING.md` focused on structural deviations and this
   file focused on live proof status.
6. Once edges pass, run `prove-exact --stage all` to close vertices and network.
