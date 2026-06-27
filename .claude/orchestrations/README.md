# Orchestrations

Multi-agent coordination artifacts for complex, multi-phase work in slavv2python.

## Overview

Orchestrations coordinate multiple agents working toward a common goal. Each orchestration has a directory containing briefings, progress tracking, handoffs, and synthesis documents.

## Directory Structure

```
orchestrations/
├── active/              # Current orchestrations
│   ├── orchestrator/    # Main project orchestration
│   └── sub_orch_*/      # Sub-orchestrations
│
└── completed/           # Archived successful orchestrations
```

## Orchestration Artifacts

### Standard Files

| File | Purpose | Owner |
|------|---------|-------|
| `BRIEFING.md` | Mission, identity, workflow, constraints | Orchestrator |
| `SCOPE.md` | Work items, milestones, acceptance criteria | Orchestrator |
| `progress.md` | Current status, checklist, decisions | Orchestrator |
| `handoff.md` | Completion summary for successor | Departing orchestrator |
| `synthesis.md` | Learnings, patterns, recommendations | Orchestrator |

### Optional Files

- `team-roster.md` - Sub-agent tracking
- `decisions.md` - Key decisions log
- `iterations/` - Iteration history if complex

## Orchestration Patterns

### 1. Iteration Loop
**Pattern**: Explorer → Worker → Reviewer → Challenger → Auditor → Gate  
**Use case**: Achieve exact numerical parity or strict acceptance criteria  
**Example**: `self_vertices_parity/`

**Phases**:
1. **Explore**: Investigate problem space
2. **Work**: Implement solution
3. **Review**: Verify correctness
4. **Challenge**: Test edge cases
5. **Audit**: Binary pass/fail gate

### 2. Pipeline Decomposition
**Pattern**: Parallel workers on independent modules  
**Use case**: Implement multiple independent features  
**Example**: Multi-stage pipeline work

**Phases**:
1. **Decompose**: Break into independent work items
2. **Dispatch**: Assign to parallel workers
3. **Integrate**: Merge results

### 3. Sequential Milestones
**Pattern**: Linear progression through milestones  
**Use case**: Dependent work where each stage builds on previous  
**Example**: Phase 1 → Phase 2 → Phase 3 certification

**Phases**:
1. **Milestone 1**: Complete with gate
2. **Milestone 2**: Blocked until M1 complete
3. **Milestone 3**: Blocked until M2 complete

## Creating an Orchestration

### 1. Initialize Structure

```bash
mkdir .claude/orchestrations/active/<orchestration_name>
cd .claude/orchestrations/active/<orchestration_name>
```

### 2. Create BRIEFING.md

```markdown
# BRIEFING — <timestamp>

## Mission
High-level goal...

## Identity
- Archetype: orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: <path>
- Parent: <parent_id>

## Workflow
- Pattern: Iteration Loop / Pipeline / Sequential
- Scope document: SCOPE.md
- Succession: At N spawns, write handoff.md

## Constraints
- DO NOT CHEAT
- Never reuse subagents after handoff
- Binary gate on acceptance

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|-------|------|-----------|--------|---------|
| ...   | ...  | ...       | ...    | ...     |
```

### 3. Create SCOPE.md

```markdown
# Scope — <orchestration_name>

## Work Items
1. Item 1 [status]
2. Item 2 [status]

## Milestones
| # | Name | Dependencies | Status |
|---|------|-------------|--------|
| M1 | ... | none | in-progress |
| M2 | ... | M1 | planned |

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
```

### 4. Track Progress

Update `progress.md` regularly:
```markdown
# Progress Tracking

Last visited: <timestamp>

## Current Status
- [x] Completed item
- [ ] Active item
- [ ] Pending item

## Recent Decisions
- Decision 1: rationale
- Decision 2: rationale

## Blockers
- Blocker 1: impact
```

## Lifecycle Management

### Active Phase
1. Keep `progress.md` updated daily
2. Document decisions in BRIEFING or decisions.md
3. Track sub-agent status in team roster
4. Update SCOPE.md when requirements change

### Completion
1. Write `handoff.md` if successor needed
2. Write `synthesis.md` with learnings
3. Extract reusable patterns to `skills/`
4. Move to `completed/` or `archive/`
5. Update `.claude/INDEX.md`

### Archive Criteria

Move to archive when:
- ✅ All acceptance criteria met
- ✅ Synthesis document written
- ✅ Learnings extracted to skills
- ✅ No active sub-agents remain
- ✅ Successor spawned (if needed)

OR

- ❌ Work abandoned (document why)
- ❌ Superseded by new approach (link to replacement)

## Best Practices

### Do
✅ Keep orchestrations focused on single high-level goal  
✅ Document decisions and rationale  
✅ Update progress frequently  
✅ Archive completed work promptly  
✅ Extract reusable patterns to skills  
✅ Maintain clear succession path

### Don't
❌ Create nested orchestrations more than 2 levels deep  
❌ Reuse sub-agents after handoff  
❌ Mix orchestration artifacts with code  
❌ Let orchestrations grow indefinitely  
❌ Skip synthesis/handoff documentation  
❌ Keep stale orchestrations in active/

## Succession Pattern

When an orchestration reaches spawn limit (typically 16):

1. Write `handoff.md`:
```markdown
# Handoff — <timestamp>

## Accomplished
- Item 1
- Item 2

## Remaining Work
- Item 3
- Item 4

## Key Learnings
- Learning 1
- Learning 2

## Successor Instructions
Continue with...
```

2. Spawn successor:
```
spawning sub_orch_<name>_2
```

3. Successor reads predecessor's handoff
4. Archive predecessor's directory

## Integration with Agents

Orchestrations use agents from `agents/`:
- **Specialists** for domain work (parity, docs, testing)
- **Explorers** for investigation
- **Workers** for implementation
- **Reviewers** for verification
- **Challengers** for adversarial testing
- **Auditors** for binary gates

## References

- [.claude/INDEX.md](../INDEX.md) - Complete catalog
- [.claude/agents/](../agents/README.md) - Agent definitions
- [docs/TODO.md](../../docs/TODO.md) - Project tasks

---

**Last Updated**: 2026-06-09
