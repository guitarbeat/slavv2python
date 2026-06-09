# Prompts

Reusable prompt templates for common workflows.

## Overview

Prompts are template workflows that can be invoked with specific context. They provide standardized approaches to recurring tasks without the full structure of an agent definition.

## Available Prompts

### Implement Python Change
**File**: `implement-python-change.prompt.md`  
**Purpose**: Standard workflow for implementing Python code changes

**When to use**:
- Adding new features
- Fixing bugs
- Refactoring existing code

**Workflow**:
1. Read impacted modules and tests
2. Make minimal targeted changes
3. Run affected tests
4. Format and lint
5. Run full suite if cross-boundary
6. Report changes and validation

### Parity Experiment
**File**: `parity-experiment.prompt.md`  
**Purpose**: Run MATLAB parity comparison workflow

**When to use**:
- Testing parity improvements
- Validating exact proof results
- Iterating on parity fixes

**Workflow**:
1. Check for active crop rerun
2. Run preflight check
3. Execute prove-exact comparison
4. Analyze discrepancies
5. Report match rate and blockers

### Regression Gate
**File**: `regression-gate.prompt.md`  
**Purpose**: Full quality gate before significant commits

**When to use**:
- Before merging to main
- After major refactoring
- Before releasing

**Workflow**:
1. Compile all modules
2. Format check
3. Lint check
4. Type check
5. Run test suite
6. Generate validation report

## Prompt Template

```markdown
---
name: "Prompt Name"
description: "When and why to use this prompt"
keywords: [keyword1, keyword2]
---

# Prompt Name

## Purpose
Why this workflow exists...

## When to Use
Specific scenarios...

## Prerequisites
What must be ready...

## Steps
1. Step 1: action
2. Step 2: action
3. Step 3: action

## Output
What gets produced...

## Validation
How to verify success...
```

## Creating New Prompts

### 1. Identify Recurring Workflow

**Good candidates for prompts**:
- Standard development workflows (implement, test, commit)
- Validation sequences (quality gate, parity check)
- Release procedures (build, test, deploy)

**Bad candidates for prompts**:
- One-off tasks (just execute directly)
- Complex multi-phase work (use orchestration)
- Domain-specific expertise (use agent)

### 2. Define Clear Steps

```markdown
## Steps

1. **Read context**: Load impacted modules
   ```bash
   # Read these files first
   ```

2. **Make changes**: Edit with constraints
   - Keep changes minimal
   - Preserve compatibility

3. **Validate**: Run checks
   ```bash
   pytest tests/unit/
   ruff check slavv_python/
   ```

4. **Report**: Summarize outcome
   - Files changed
   - Tests passed
   - Remaining work
```

### 3. Specify Validation

```markdown
## Validation

Success criteria:
- [ ] All tests pass
- [ ] Linter reports no errors
- [ ] Type checker passes
- [ ] Behavior matches requirements

Failure handling:
- If tests fail: Review test output, fix, retry
- If lint fails: Run `ruff check --fix`, verify
- If types fail: Add missing hints, verify
```

### 4. Document Prerequisites

```markdown
## Prerequisites

- [ ] Virtual environment activated
- [ ] Dependencies installed: `pip install -e ".[app,workspace]"`
- [ ] No uncommitted changes (or stash first)
- [ ] On appropriate branch
```

## Prompts vs Agents vs Instructions

### Prompts
- **Structure**: Template workflow
- **Activation**: Manual invocation with context
- **Purpose**: Guide standardized task execution
- **Example**: "Run regression gate"

### Agents
- **Structure**: Full AI assistant with expertise
- **Activation**: Agent invocation with task
- **Purpose**: Execute tasks with domain knowledge
- **Example**: "MATLAB Parity Specialist fixes edges"

### Instructions
- **Structure**: Conditional constraints
- **Activation**: Automatic when pattern matches
- **Purpose**: Enforce local rules during work
- **Example**: "Run parity tests for parity code"

## Usage Patterns

### Direct Invocation
```
/use prompt: implement-python-change
Context: Fix vertex energy calculation bug
Files: slavv_python/processing/stages/vertices/detection.py
```

### From Agent
Agents can reference prompts:
```markdown
## Approach
Follow the standard "implement-python-change" prompt workflow:
1. Read impacted modules
2. Make targeted changes
...
```

### In Orchestration
Orchestrations can standardize sub-agent workflows:
```markdown
## Worker Instructions
Each worker follows "implement-python-change" prompt for their assigned module.
```

## Best Practices

### Do
✅ Keep prompts focused on single workflow  
✅ Define clear success criteria  
✅ Provide concrete examples  
✅ List prerequisites explicitly  
✅ Specify validation commands  
✅ Update as tools evolve

### Don't
❌ Make prompts too long (>50 lines)  
❌ Include domain expertise (use agent)  
❌ Duplicate existing agent workflows  
❌ Hard-code project specifics  
❌ Skip validation steps

## Maintenance

### Regular Updates

**After tool changes**:
- Update command syntax
- Adjust validation steps
- Fix deprecated flags

**After workflow improvements**:
- Incorporate learnings
- Add discovered edge cases
- Refine success criteria

### Deprecation

When deprecating a prompt:
1. Mark as deprecated in frontmatter
2. Link to replacement (if any)
3. Keep for 1 release cycle
4. Move to `archive/deprecated-prompts/`
5. Update INDEX.md

## Common Workflows

### Development Cycle
```
implement-python-change
  ↓
regression-gate (if cross-boundary)
  ↓
git commit (follows git-commit-message rule)
  ↓
PR creation (follows make-pr-easy-to-review skill)
```

### Parity Work
```
parity-experiment (run prove-exact)
  ↓
implement-python-change (fix discrepancies)
  ↓
parity-experiment (verify improvement)
  ↓
regression-gate (before merge)
```

### Release Process
```
regression-gate (full validation)
  ↓
(external) build artifacts
  ↓
(external) deploy to environment
  ↓
(external) smoke tests
```

## Prompt Library Organization

### By Stage
- **Development**: `implement-python-change`
- **Validation**: `regression-gate`, `parity-experiment`
- **Release**: (future: `prepare-release`, `publish-artifacts`)

### By Scope
- **Code**: `implement-python-change`
- **Quality**: `regression-gate`
- **Parity**: `parity-experiment`

## Integration

### With Skills
Prompts may invoke skills:
```markdown
## Steps
3. **Review quality**: Use "thermo-nuclear-code-quality-review" skill
4. **Run tests**: Use "run-smoke-tests" skill
```

### With Tools
Prompts reference tool commands:
```bash
# Lint
ruff check slavv_python tests

# Type check  
mypy

# Test
pytest -m "unit or integration"
```

### With Documentation
Prompts link to docs:
```markdown
## References
- [Testing Guide](../../tests/README.md)
- [Python Naming](../../docs/reference/workflow/PYTHON_NAMING_GUIDE.md)
```

## Examples

### Minimal Prompt
```markdown
---
name: "Quick Lint Fix"
description: "Fix linting errors quickly"
---

# Quick Lint Fix

## Steps
1. Run: `ruff check --fix slavv_python tests`
2. Review changes
3. Run: `ruff format slavv_python tests`

## Validation
- [ ] `ruff check` passes
```

### Full Workflow Prompt
```markdown
---
name: "Feature Implementation"
description: "Complete workflow for new feature"
---

# Feature Implementation

## Prerequisites
- [ ] Feature spec reviewed
- [ ] Design approved
- [ ] Branch created

## Steps
1. Read: Impacted modules and tests
2. Implement: Feature with tests
3. Validate: Run test suite
4. Document: Update docs if needed
5. Review: Self-review checklist

## Output
- Working feature
- Test coverage
- Updated documentation

## Validation
- [ ] Tests pass
- [ ] Linter clean
- [ ] Types verify
- [ ] Docs updated
```

## References

- [.agents/INDEX.md](../INDEX.md) - Complete catalog
- [.agents/agents/](../agents/README.md) - Agent definitions
- [.agents/skills/](../skills/README.md) - Skills library
- [CONTRIBUTING.md](../../docs/CONTRIBUTING.md) - Development guide

---

**Last Updated**: 2026-06-09
