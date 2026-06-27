# Rules

Global rules that always apply across the entire repository.

## Overview

Rules are always-active constraints that enforce repository-wide standards and conventions. Unlike instructions (which apply conditionally), rules apply to all work.

## Rule Types

### Always-Apply Rules
Apply continuously during all work.

**Format**: `*.md` with `alwaysApply: true` frontmatter

### Scene Rules
Apply in specific scenarios (git commit, PR creation, etc.)

**Format**: `*.md` with `scene: <scenario>` frontmatter

## Available Rules

### Git Commit Message
**File**: `git-commit-message.md`  
**Scene**: `git_message`  
**Purpose**: Enforce conventional commit format

**Format**:
```
<type>(<scope>): <short summary>

<optional body>
```

**Types**: `fix`, `feat`, `refactor`, `docs`, `test`, `chore`, `parity`, `perf`

**Scopes**: `energy`, `vertices`, `edges`, `network`, `engine`, `cli`, `parity`, etc.

**Examples**:
```
fix(edges): correct frontier insertion priority for hub vertices
parity(edges): align candidate filtering to match MATLAB (88.7%→92%)
docs: update all path references after package reorganization
```

### No Inline Imports
**File**: `no-inline-imports.mdc`  
**Scene**: Always  
**Purpose**: Prevent import statements inside functions

**Rationale**: 
- Makes dependencies explicit at module level
- Improves code clarity and maintainability
- Prevents circular import issues hidden in runtime

**Example**:
```python
# ❌ Bad
def process():
    import numpy as np
    return np.array([1, 2, 3])

# ✅ Good
import numpy as np

def process():
    return np.array([1, 2, 3])
```

### Python Quality Verification
**File**: `python-quality-verification.mdc`  
**Scene**: Always  
**Purpose**: Enforce Python code quality standards

**Standards**:
- Type hints on function signatures
- Docstrings for public functions/classes
- Proper error handling (not bare `except:`)
- Descriptive variable names
- No unused imports
- Prefer explicit over implicit

### TypeScript Exhaustive Switch
> **Removed** — This rule was for TypeScript projects. It has been deleted from this Python-only repository.



**Pattern**:
```typescript
// ✅ Good - exhaustive with default
switch (value) {
    case 'a': return 1;
    case 'b': return 2;
    default: 
        const _exhaustive: never = value;
        throw new Error(`Unhandled case: ${value}`);
}
```

## Rule Template

```markdown
---
alwaysApply: true
scene: <optional_scene>
---

# Rule Name

## Purpose
Why this rule exists...

## Standard
What the rule requires...

## Examples
Good and bad examples...

## Exceptions
When rule doesn't apply (if any)...
```

## Creating New Rules

### 1. Identify Need

**Good candidates for rules**:
- Repository-wide conventions (commit format, code style)
- Universal constraints (no secrets in code, no print() in library)
- Critical guardrails (test before merge, type hints required)

**Bad candidates for rules**:
- File-specific conventions (use instructions/ instead)
- Task-specific workflows (use agents/ instead)
- Temporary constraints (document in code instead)

### 2. Write Clear Standard

```markdown
## Standard

All Python library code must:
1. Have type hints on public functions
2. Include docstrings for public APIs
3. Use logging, not print()
4. Handle errors explicitly
```

### 3. Provide Examples

Show both correct and incorrect:
```markdown
## Examples

### ❌ Bad
```python
def process(data):
    print("Processing...")
    return data + 1
```

### ✅ Good
```python
import logging

logger = logging.getLogger(__name__)

def process(data: int) -> int:
    """Process data by incrementing."""
    logger.info("Processing data")
    return data + 1
```
```

### 4. Document Exceptions

If rule has exceptions, be explicit:
```markdown
## Exceptions

- Scripts in `scripts/` may use print() for CLI output
- Test code doesn't require docstrings
- Private functions (leading `_`) may omit type hints
```

## Rules vs Instructions vs Agents

### Rules
- **Scope**: Always apply globally
- **Activation**: Automatic always
- **Purpose**: Enforce universal standards
- **Example**: "Use conventional commit format"

### Instructions
- **Scope**: Specific file patterns or scenarios
- **Activation**: Automatic when pattern matches
- **Purpose**: Enforce local constraints
- **Example**: "Run parity tests for parity changes"

### Agents
- **Scope**: Complete tasks end-to-end
- **Activation**: Explicit invocation
- **Purpose**: Execute workflows
- **Example**: "Fix parity discrepancies"

## Best Practices

### Do
✅ Keep rules universally applicable  
✅ Document clear rationale  
✅ Provide concrete examples  
✅ Specify exceptions explicitly  
✅ Update as standards evolve

### Don't
❌ Create rules for specific modules (use instructions)  
❌ Make rules too prescriptive (allow reasonable judgment)  
❌ Duplicate code style that linters enforce  
❌ Add rules without team agreement  
❌ Leave rules ambiguous

## Enforcement

### Automated Enforcement
Some rules are enforced by tools:
- **no-inline-imports**: Ruff linter
- **python-quality-verification**: mypy, ruff
- **git-commit-message**: pre-commit hook (optional)

### Manual Review
Other rules require human judgment:
- Commit message quality
- Code organization
- Design decisions

## Maintenance

### Regular Review

**Quarterly**:
1. Review rule effectiveness
2. Check for outdated standards
3. Consolidate overlapping rules
4. Update examples with current patterns

**After major changes**:
- Update examples if API changes
- Adjust exceptions if structure changes
- Sync with linter configurations

### Deprecation

When deprecating a rule:
1. Document why in rule file
2. Move to `archive/deprecated-rules/`
3. Update dependent instructions/agents
4. Announce to team
5. Update INDEX.md

## Integration

### With Linters
Rules should align with linter config:
```toml
# pyproject.toml
[tool.ruff]
select = ["E", "F", "I"]  # Enforces import ordering

# Corresponding rule:
# no-inline-imports.mdc
```

### With Pre-commit Hooks
Critical rules in `.pre-commit-config.yaml`:
```yaml
- repo: local
  hooks:
    - id: commit-msg-format
      name: Conventional Commit Format
      # References: .claude/rules/git-commit-message.md
```

### With Documentation
Rules referenced in:
- `AGENTS.md` - Agent constraints
- `CONTRIBUTING.md` - Contributor guide
- `tests/README.md` - Test standards

## Rule Hierarchy

When rules conflict (rare), priority order:
1. **Safety rules** - No secrets, proper error handling
2. **Quality rules** - Type hints, docstrings, tests
3. **Convention rules** - Commit format, naming
4. **Style rules** - Code formatting (defer to ruff)

## Examples by Category

### Code Quality
- `python-quality-verification.mdc` - General quality
- `no-inline-imports.mdc` - Import discipline

### Process
- `git-commit-message.md` - Commit format

### Future Candidates
- No print() in library code (logging only)
- Explicit encoding for file operations
- Prefer pathlib over os.path

## References

- [.claude/INDEX.md](../INDEX.md) - Complete catalog
- [.claude/instructions/](../instructions/README.md) - Scoped instructions
- [CONTRIBUTING.md](../../docs/CONTRIBUTING.md) - Contribution guide
- [pyproject.toml](../../pyproject.toml) - Tool configuration

---

**Last Updated**: 2026-06-09
