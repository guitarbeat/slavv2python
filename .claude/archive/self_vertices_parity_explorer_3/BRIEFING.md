# BRIEFING — 2026-06-08T17:41:00Z

## Mission
Analyze the `vertices` stage on `180709_E_crop_M` for exact parity with MATLAB truth.

## 🔒 My Identity
- Archetype: Explorer
- Roles: Read-only investigator
- Working directory: d:\2P_Data\Aaron\slavv2python\.agents\self_vertices_parity_explorer_3
- Original parent: 89d02147-ad78-4172-9773-75165fd69f08
- Milestone: R1. Vertices Parity

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Must follow the Handoff Protocol

## Current Parent
- Conversation ID: 89d02147-ad78-4172-9773-75165fd69f08
- Updated: not yet

## Investigation State
- **Explored paths**: `slavv_python/processing/stages/vertices/`, `external/Vectorization-Public/source/get_vertices_V200.m`, `external/Vectorization-Public/source/vectorize_V200.m`
- **Key findings**: Python uses global linear-index tie-breaking due to `np.lexsort` in `sort_vertex_order`, whereas MATLAB stably sorts and preserves chunk-concatenation order for global ties.
- **Unexplored areas**: None.

## Key Decisions Made
- Diagnosed the tie-breaking bug using static code analysis without waiting for the blocked `prove-exact` run.

## Artifact Index
- `handoff.md` — Handoff report with findings and fix strategy
