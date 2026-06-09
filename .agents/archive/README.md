# Archive

Historical orchestrations and deprecated agent work.

## Overview

This directory contains completed, abandoned, or superseded orchestrations that are kept for historical reference but are no longer active.

## Contents

### Vertices Parity Orchestration (2026-06-08)

**Main Orchestration**: `self_vertices_parity/`
- **Goal**: Achieve 100% exact numerical parity for vertices stage
- **Pattern**: Iteration loop (Explorer → Worker → Reviewer → Challenger → Auditor)
- **Status**: Superseded by direct implementation approach
- **Outcome**: Spawned multiple sub-agents before pattern was refined

**Sub-Agents**:
- `self_vertices_parity_explorer_1/` - Initial problem space investigation
- `self_vertices_parity_explorer_2/` - Second investigation iteration
- `self_vertices_parity_explorer_3/` - Third investigation iteration
- `self_vertices_parity_worker_1/` - Implementation attempt
- `self_vertices_parity_reviewer_1/` - Review phase 1
- `self_vertices_parity_reviewer_2/` - Review phase 2
- `self_vertices_parity_challenger_1/` - Challenge phase 1
- `self_vertices_parity_challenger_2/` - Challenge phase 2
- `self_vertices_parity_auditor_1/` - Audit gate

**Key Learnings**:
- Iteration loop pattern needs clearer exit criteria
- Sub-agent handoff protocol is critical
- Binary audit gates prevent endless iteration
- Direct implementation sometimes more efficient than heavy orchestration

### Sentinel Orchestration

**Location**: `sentinel/`
- **Goal**: Monitor and report on repository health
- **Status**: Completed/deprecated
- **Outcome**: Monitoring patterns extracted to skills

## Archive Structure

```
archive/
├── README.md                               # This file
│
├── self_vertices_parity/                   # Main orchestration
│   ├── BRIEFING.md
│   ├── SCOPE.md
│   ├── progress.md
│   └── synthesis.md
│
├── self_vertices_parity_*/                 # Sub-agents (9 total)
│   ├── BRIEFING.md
│   ├── progress.md
│   └── handoff.md
│
└── sentinel/                               # Monitoring orchestration
    ├── BRIEFING.md
    └── handoff.md
```

## Accessing Archived Content

### When to Read Archives

✅ **Do read archives when**:
- Understanding past approaches to current problems
- Learning from previous orchestration patterns
- Extracting patterns for new orchestrations
- Debugging similar issues

❌ **Don't read archives for**:
- Current work status (use `docs/TODO.md`)
- Active orchestrations (see `orchestrations/active/`)
- Implementation guidance (use maintained docs)

### Finding Relevant Archives

1. **By topic**: Search for keywords in archive file names
2. **By date**: Check BRIEFING.md timestamps
3. **By pattern**: Look at workflow patterns in BRIEFING.md
4. **By outcome**: Read synthesis.md or handoff.md

## Archive Retention Policy

### Keep Archives That
- Document significant learnings
- Show pattern evolution
- Contain unique problem-solving approaches
- Have detailed synthesis documents

### Consider Removing Archives That
- Have no synthesis/handoff documents
- Are duplicates or similar to other archives
- Contain only boilerplate without substance
- Are superseded by better-documented work

### Regular Maintenance

**Quarterly Review**:
1. Scan archives for extractable patterns
2. Move patterns to skills/ if reusable
3. Consolidate similar archives
4. Update this README with key learnings
5. Remove archives with no historical value

## Extracted Patterns

Patterns from these archives now live in:

### From self_vertices_parity
- **Pattern**: Iteration loop with binary gates
- **Lesson**: Clear exit criteria prevent endless cycles
- **Skill**: Not yet extracted (candidate for systematic-debugging)

### From sentinel
- **Pattern**: Continuous monitoring with reporting
- **Lesson**: Automated checks better than manual reviews
- **Skill**: Extracted to monitoring-related skills

## Key Learnings Summary

### Orchestration Design
1. **Spawn limits matter**: 16 spawns before succession prevents bloat
2. **Binary gates required**: Prevent endless iteration
3. **Handoffs are critical**: Document for successors
4. **Synthesis is valuable**: Extract learnings promptly

### Agent Patterns
1. **Exploration depth**: Too many explorers indicates unclear scope
2. **Worker parallelization**: Works for independent modules
3. **Review thoroughness**: Reviewers need clear acceptance criteria
4. **Challenger creativity**: Adversarial testing finds edge cases

### Process Improvements
1. Start with clearer acceptance criteria
2. Limit sub-agent depth (2-3 levels max)
3. Document decisions as they happen
4. Extract patterns to skills immediately after completion

## Migration Notes

### Moving Archives Here

When archiving an orchestration:

```bash
# From active/
mv .agents/orchestrations/active/<name> .agents/archive/

# Add entry to this README
# Extract patterns to skills if applicable
# Update .agents/INDEX.md
```

### Archive Document Requirements

Each archived orchestration should have:
- ✅ BRIEFING.md (what was attempted)
- ✅ progress.md or handoff.md (what happened)
- ✅ synthesis.md (what was learned) - **preferred**
- ⚠️ If missing synthesis, add brief notes to this README

## References

- [Active Orchestrations](../orchestrations/active/) - Current work
- [Completed Orchestrations](../orchestrations/completed/) - Successful completions
- [Skills Library](../skills/) - Extracted reusable patterns
- [Project TODO](../../docs/TODO.md) - Current tasks

---

**Last Updated**: 2026-06-09  
**Last Archive Addition**: self_vertices_parity (2026-06-08)
