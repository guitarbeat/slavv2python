# Agent Definitions

AI agent definitions for specialized tasks in the slavv2python repository.

## Overview

Agents are AI assistants with specific expertise, tools, and constraints. Each agent is defined in a `.agent.md` file with frontmatter configuration and body instructions.

## Agent Template

```markdown
---
name: "Agent Name"
description: "Brief description with keywords for discoverability"
tools: [read, search, edit, execute, todo, agent]
agents: [SubAgent1, SubAgent2]
user-invocable: true
---

# Agent Instructions

Your job is...

## Context
Files to read before starting...

## Constraints
What not to do...

## Approach
Step-by-step workflow...

## Output Format
What to return...
```

## Available Agents

### Specialist Agents

#### MATLAB Parity Specialist
**File**: `matlab-parity-specialist.agent.md`  
**Keywords**: parity, MATLAB, comparison, edges, watershed, exact proof  
**Purpose**: Preserve and improve MATLAB-to-Python parity behavior  
**Key Modules**: `pipeline/edges/`, `analytics/parity/`

**When to use**:
- Changing parity/comparison workflows
- Fixing exact proof discrepancies
- Working on watershed or edge discovery
- Modifying oracle/run-root layout

#### Docs Link Auditor
**File**: `docs-link-auditor.agent.md`  
**Keywords**: docs audit, link check, path drift, documentation consolidation  
**Purpose**: Audit markdown documentation for accuracy and completeness  

**When to use**:
- Checking for broken documentation links
- Finding stale path references after refactoring
- Identifying duplicated content
- Validating documentation against code

#### Python Refactor + Tests
**File**: `python-refactor-tests.agent.md`  
**Keywords**: refactor, tests, code quality  
**Purpose**: Refactor Python code while maintaining/adding test coverage  

**When to use**:
- Code restructuring with tests
- Extracting reusable patterns
- Improving code quality

### Utility Agents

#### CI Watcher
**File**: `agents/ci-watcher.md`  
**Purpose**: Monitor CI pipeline health  

#### Thermo-Nuclear Code Quality Review
**File**: `agents/thermo-nuclear-code-quality-review.md`  
**Purpose**: Deep code quality analysis  

## Creating a New Agent

### 1. Choose Agent Type

**Specialist Agent**
- Domain-focused (parity, docs, testing)
- Deep expertise in specific area
- Has context files to read first
- May delegate exploration to sub-agents

**Utility Agent**
- General-purpose helper
- Lighter constraints
- Quick focused tasks

### 2. Define Configuration

```yaml
---
name: "Your Agent Name"
description: "Clear description with keywords: keyword1, keyword2, keyword3"
tools: [read, search, edit, execute]  # Choose tools needed
agents: []                              # Can invoke other agents
user-invocable: true                   # Can be manually invoked
---
```

### 3. Write Instructions

Include:
- **Context**: What to read before starting
- **Key Module Locations**: Where relevant code lives
- **Constraints**: What to avoid or preserve
- **Approach**: Step-by-step workflow
- **Output Format**: What to return

### 4. Test and Iterate

- Test agent with typical tasks
- Refine instructions based on outcomes
- Update context files as codebase evolves

## Best Practices

### Do
✅ Use clear, actionable instructions  
✅ Reference maintained documentation  
✅ Specify validation commands  
✅ List constraints explicitly  
✅ Provide step-by-step approach  
✅ Define expected output format

### Don't
❌ Make agents too broad or generic  
❌ Reference stale file paths  
❌ Skip validation requirements  
❌ Assume context without reading  
❌ Duplicate instructions across agents  
❌ Create overlapping agent purposes

## Agent Invocation

### By User
```
@matlab-parity-specialist
/investigate edges watershed mismatch
```

### By Agent
Agents can invoke other agents if configured:
```yaml
agents: [Explore, "Python Refactor + Tests"]
```

## Maintenance

### Regular Updates
- Update context file paths after refactoring
- Refresh module locations when structure changes
- Add new constraints as patterns emerge
- Update validation commands with new tools

### Deprecation
When an agent is no longer needed:
1. Move to `archive/deprecated-agents/`
2. Document reason for deprecation
3. Update INDEX.md
4. Provide migration path if applicable

## Integration with Instructions

Agents and Instructions work together:
- **Agents**: High-level task execution with context
- **Instructions**: Low-level rules applied to specific file patterns

Example:
- `matlab-parity-specialist.agent.md` handles parity tasks
- `parity-safe-change.instructions.md` enforces constraints during editing

## References

- [.claude/INDEX.md](../INDEX.md) - Complete catalog
- [.claude/instructions/](../instructions/README.md) - Scoped instructions
- [.claude/skills/](../skills/README.md) - Reusable skills

---

**Last Updated**: 2026-06-09
