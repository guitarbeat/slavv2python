# Instructions

Scoped instructions that apply conditionally based on file patterns or scenarios.

## Overview

Instructions provide targeted guidance that activates automatically when working on specific files or in specific contexts. Unlike agents (which handle entire tasks), instructions modify behavior during task execution.

## Instruction Types

### File-Pattern Instructions
Apply when editing files matching specific patterns.

**Format**: `*.instructions.md` with `applyTo` frontmatter

### Scene Instructions
Apply in specific scenarios (git commit, PR creation, etc.)

**Format**: See rules/ for scene-based guidance

## Available Instructions

### Parity-Safe Change
**File**: `parity-safe-change.instructions.md`  
**ApplyTo**: `slavv_python/{analytics/parity,engine/state}/**/*.py`  
**Purpose**: Enforce staged layout compatibility when changing parity code

**Enforces**:
- Preserve staged run layout semantics (`01_Input/`, `02_Output/`, etc.)
- Run parity-specific tests for parity changes
- Favor additive compatibility over breaking changes
- Add regression tests for parity output changes

### Doc Path Hygiene
**File**: `doc-path-hygiene.instructions.md`  
**ApplyTo**: `docs/**/*.md`  
**Purpose**: Maintain accurate file paths in documentation

**Enforces**:
- Verify paths exist before committing
- Update cross-references when moving files
- Use relative paths within docs/
- Check code examples reference actual files

### Tests Placement
**File**: `tests-placement.instructions.md`  
**ApplyTo**: `tests/**/*.py`  
**Purpose**: Ensure tests go in correct ownership folders

**Enforces**:
- Unit tests under `tests/unit/<owner>/`
- Integration tests under `tests/integration/`
- Parity tests under `tests/integration/parity/`
- Regression marker for files with "regression" in name

## Instruction Template

```markdown
---
description: "When and why this instruction applies"
applyTo: "file/pattern/**/*.py"
---

# Instruction Name

## Scope
When these instructions apply...

## Requirements
What must be true...

## Validation
How to verify compliance...

## References
- [Relevant doc 1](path/to/doc1.md)
- [Relevant doc 2](path/to/doc2.md)
```

## Creating New Instructions

### 1. Identify Scope

**Good candidates for instructions**:
- File-specific conventions (test placement, import style)
- Module-specific constraints (parity compatibility, API stability)
- Domain-specific validation (run layout, checkpoint format)

**Bad candidates for instructions**:
- General coding standards (use rules/ instead)
- Full task workflows (use agents/ instead)
- One-time fixes (just document in code)

### 2. Write Clear Constraints

```markdown
## Validation Expectations

When changing parity comparison behavior:
- Run: `pytest tests/integration/parity/`
- Run: `pytest tests/unit/analysis/ -k parity`

When changes cross module boundaries:
- Run: `pytest -m "unit or integration"`
```

### 3. Provide Context

Link to:
- Relevant maintained docs
- Module ownership info
- Architecture decision records (ADRs)
- Code examples

### 4. Test Activation

Verify instruction activates when:
- Editing matching file patterns
- In specified scenarios
- With expected constraints applied

## Instructions vs Rules vs Agents

### Instructions
- **Scope**: Specific file patterns or scenarios
- **Activation**: Automatic when pattern matches
- **Purpose**: Modify behavior during task execution
- **Example**: "When editing parity code, run parity tests"

### Rules
- **Scope**: Always apply globally
- **Activation**: Automatic always
- **Purpose**: Enforce repository-wide standards
- **Example**: "Use conventional commit format"

### Agents
- **Scope**: Complete tasks end-to-end
- **Activation**: Explicit invocation
- **Purpose**: Execute workflows with context
- **Example**: "Fix parity discrepancies in edges stage"

## Best Practices

### Do
✅ Keep instructions focused on single concern  
✅ Provide specific validation commands  
✅ Link to authoritative documentation  
✅ Update when file patterns change  
✅ Test that pattern matching works

### Don't
❌ Make instructions too broad  
❌ Duplicate global rules  
❌ Include full implementation guidance  
❌ Reference stale file paths  
❌ Create conflicting instructions

## Maintenance

### Regular Updates

**After refactoring**:
- Update `applyTo` patterns if paths changed
- Verify referenced docs still exist
- Test pattern matching

**When standards change**:
- Update validation commands
- Sync with related rules/agents
- Document in CHANGELOG

### Deprecation

When an instruction is no longer needed:
1. Check for dependent agents/workflows
2. Move to `archive/deprecated-instructions/`
3. Update INDEX.md
4. Remove references from agents

## Integration with Other Components

### With Agents
Agents may reference instructions:
```markdown
## Constraints
Follow parity-safe-change instructions when editing
analytics/parity/ modules.
```

### With Rules
Instructions provide specificity; rules provide broad guidance:
- **Rule**: "Write tests for new features"
- **Instruction**: "Put unit tests under tests/unit/<owner>/"

### With Skills
Skills may reinforce instructions:
- **Instruction**: "Run parity tests for parity changes"
- **Skill**: "systematic-testing" provides test execution framework

## Examples

### File Pattern Instruction
```yaml
---
applyTo: "slavv_python/processing/stages/**/*.py"
---
When changing pipeline stages, run stage-specific tests first.
```

### Scenario Instruction
```yaml
---
scene: "pre_commit"
---
Before committing, run: ruff check && pytest -m unit
```

## References

- [.agents/INDEX.md](../INDEX.md) - Complete catalog
- [.agents/rules/](../rules/README.md) - Global rules
- [.agents/agents/](../agents/README.md) - Agent definitions
- [tests/README.md](../../tests/README.md) - Test organization

---

**Last Updated**: 2026-06-09
