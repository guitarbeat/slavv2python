# .agents Directory

This directory contains AI agent definitions, instructions, skills, and orchestration artifacts for the slavv2python repository.

## 📁 Directory Structure

```
.claude/
├── README.md                          # This file
├── INDEX.md                           # Quick reference catalog
├── HANDOFF.md                         # Current parity successor brief
│
├── agents/                            # Active agent definitions
│   ├── *.agent.md                     # Agent definition files
│   └── README.md                      # Agent catalog and usage guide
│
├── instructions/                      # Scoped instructions
│   ├── *.instructions.md              # Conditional instructions by file pattern
│   └── README.md                      # Instructions guide
│
├── rules/                             # Global rules
│   ├── *.md / *.mdc                   # Always-applied rules
│   └── README.md                      # Rules catalog
│
├── prompts/                           # Reusable prompt templates
│   ├── *.prompt.md                    # Prompt templates
│   └── README.md                      # Prompts catalog
│
├── skills/                            # Reusable skills library
│   ├── */SKILL.md                     # Skill definitions
│   └── README.md                      # Skills catalog
│
└── orchestrations/                    # Orchestration work artifacts
    ├── active/                        # Current orchestrations
    ├── completed/                     # Archived successful orchestrations
    └── README.md                      # Orchestration guide
```

## 🎯 Quick Start

### For AI Agents
1. **Read**: Repository guidance in root `/AGENTS.md` (automatically loaded)
2. **Check**: [HANDOFF.md](HANDOFF.md), then the canonical task and status
   documents it links; `orchestrations/active/` is empty unless a genuinely
   active multi-agent effort has been started.
3. **Use**: Skills from `skills/` library as needed
4. **Follow**: Rules in `rules/` (always applied)

### For Humans
1. **Browse**: `INDEX.md` for quick overview
2. **Create**: New agents in `agents/` directory
3. **Archive**: Completed work to appropriate folders
4. **Maintain**: Keep active vs. archived separation clear

## 📖 Documentation

### Core Files
- **HANDOFF.md** - Current parity decision point and operating sequence
- **INDEX.md** - Catalog of all agents, instructions, rules, prompts
- **agents/README.md** - How to create and use agent definitions
- **skills/README.md** - Skills library and contribution guide
- **orchestrations/README.md** - Multi-agent orchestration patterns

### Agent Types

| Type | Purpose | Location |
|------|---------|----------|
| **Specialist** | Domain-focused work (parity, docs, testing) | `agents/*.agent.md` |
| **Orchestrator** | Multi-agent coordination | `orchestrations/active/` |
| **Utility** | General-purpose helpers | `agents/*.agent.md` |

## 🔄 Lifecycle Management

### Active Work
- Agent definitions go in `agents/`
- Active orchestrations go in `orchestrations/active/` only while they are
  genuinely active.
- Repository work state belongs in `docs/TODO.md` and
  `docs/reference/core/EXACT_PROOF_FINDINGS.md`; do not duplicate it in agent
  briefings or progress files.

### Completion
- Move successful orchestrations to `orchestrations/completed/`
- Extract learnings to `skills/` if reusable
- Archive supporting files to `archive/`
- Document in `INDEX.md`

### Cleanup Guidelines
1. Archive completed orchestrations within 1 week of completion
2. Extract reusable patterns to skills library
3. Remove stale progress files and temporary artifacts
4. Keep only active work in top-level directories

## 🎨 Naming Conventions

### Agent Files
```
<purpose>-<domain>.agent.md
```
Examples: `matlab-parity-specialist.agent.md`, `docs-link-auditor.agent.md`

### Orchestration Directories
```
<project>_<milestone>/
```
Examples: `vertices_parity/`, `energy_optimization/`

### Instruction Files
```
<scope>-<behavior>.instructions.md
```
Examples: `parity-safe-change.instructions.md`, `test-placement.instructions.md`

## 🚫 Anti-Patterns

❌ **Don't keep completed orchestrations in root**  
✅ Archive to `orchestrations/completed/` or `archive/`

❌ **Don't create numbered orchestration variants** (`_1`, `_2`, `_3`)  
✅ Use `orchestrations/active/<name>/iterations/` for iteration tracking

❌ **Don't duplicate agent definitions**  
✅ Reuse existing agents or update the canonical version

❌ **Don't leave empty directories**  
✅ Clean up or add placeholder README.md

## 📚 References

- [Root AGENTS.md](../AGENTS.md) - Repository-wide agent guidance
- [Repository TODO](../docs/TODO.md) - Active tasks
- [Documentation Index](../docs/README.md) - Full documentation map

---

**Last Updated**: 2026-06-09  
**Maintained By**: Repository maintainers
