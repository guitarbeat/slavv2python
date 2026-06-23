# .agents Quick Reference

Fast lookup guide for common tasks and questions.

---

## 🚀 I want to...

### Use an Agent
```
@matlab-parity-specialist
/investigate parity mismatch in edges
```
**See**: [INDEX.md](INDEX.md#agents) for all available agents

### Find a Skill
**Browse**: [INDEX.md](INDEX.md#skills) → Skills by category  
**Use**: Reference skill name from agents or orchestrations

### Apply an Instruction
**Automatic**: Edit files matching pattern → instruction applies  
**Manual**: Reference in agent constraints  
**See**: [instructions/README.md](instructions/README.md)

### Follow a Rule
**Automatic**: Rules always apply globally  
**Check**: [rules/README.md](rules/README.md) for all rules  
**Example**: Git commits follow `git-commit-message.md`

### Run a Prompt
```
/use prompt: implement-python-change
Context: <your context>
```
**See**: [prompts/README.md](prompts/README.md) for workflow templates

### Start an Orchestration
1. Create directory in `orchestrations/active/<name>/`
2. Write `BRIEFING.md` and `SCOPE.md`
3. Follow pattern from [orchestrations/README.md](orchestrations/README.md)

### Archive Completed Work
1. Write synthesis/handoff document
2. Move to `archive/` or `orchestrations/completed/`
3. Extract patterns to `skills/`
4. Update `INDEX.md`

---

## 🔍 Where is...?

| What | Location |
|------|----------|
| **Agent definitions** | `agents/*.agent.md` |
| **Skills library** | `skills/*/SKILL.md` |
| **Active orchestrations** | `orchestrations/active/` |
| **Completed work** | `archive/` or `orchestrations/completed/` |
| **Instructions** | `instructions/*.instructions.md` |
| **Rules** | `rules/*.md` or `rules/*.mdc` |
| **Prompt templates** | `prompts/*.prompt.md` |
| **Current handoff brief** | `HANDOFF.md` |
| **Complete catalog** | `INDEX.md` |
| **Ephemeral agent output** | `workspace/scratch/agent-tools/` (not repo root) |

---

## 🎯 Common Tasks

### Check Available Agents
```
Open: .agents/INDEX.md
Section: Agents → Specialist Agents
```

### Find Parity-Related Tools
```
Open: .agents/INDEX.md
Section: Quick Search → By Task Type → MATLAB Parity
```

### Create New Agent
```
1. Read: agents/README.md (template section)
2. Create: agents/<name>.agent.md
3. Update: INDEX.md
```

### Check Active Orchestrations
```
List: .agents/orchestrations/active/
Read: <orchestration>/progress.md
```

### Archive an Orchestration
```
1. Write: <orchestration>/synthesis.md or handoff.md
2. Move: orchestrations/active/<name> → archive/<name>
3. Update: INDEX.md
```

---

## 📋 By Role

### For AI Agents

**Starting work**:
1. Read root `AGENTS.md` (auto-loaded)
2. Check `INDEX.md` for available tools
3. Navigate to relevant subdirectory README
4. Use appropriate agents/skills/instructions

**During work**:
- Instructions apply automatically by file pattern
- Rules always apply
- Invoke agents with `@agent-name`
- Reference skills by name

### For Developers

**Understanding organization**:
```
README.md → Overview and navigation
INDEX.md → Complete catalog
Subdirectory READMEs → Detailed guides
```

**Creating content**:
```
1. Check subdirectory README for template
2. Follow naming conventions
3. Update INDEX.md
```

**Finding examples**:
```
agents/README.md → Agent examples
orchestrations/README.md → Pattern examples
archive/ → Historical examples
```

### For Orchestrators

**Starting orchestration**:
```
1. Read: orchestrations/README.md
2. Create: orchestrations/active/<name>/
3. Write: BRIEFING.md, SCOPE.md, progress.md
```

**During orchestration**:
```
Update: progress.md (regularly)
Track: Team roster in BRIEFING.md
Document: Decisions as they happen
```

**Completing orchestration**:
```
1. Write: synthesis.md or handoff.md
2. Extract: Patterns to skills/
3. Move: To archive/ or completed/
4. Update: INDEX.md
```

---

## 🔗 Key Files

| File | Purpose | When to Read |
|------|---------|-------------|
| `README.md` | Main guide | Understanding organization |
| `INDEX.md` | Complete catalog | Finding specific items |
| `IMPROVEMENT_SUMMARY.md` | Change history | Understanding evolution |
| `QUICK_REFERENCE.md` | This file | Fast lookups |
| `agents/README.md` | Agent guide | Creating/using agents |
| `orchestrations/README.md` | Orchestration guide | Multi-agent work |
| `archive/README.md` | Archive guide | Historical context |

---

## 🎨 Naming Conventions

```
Agents:          <purpose>-<domain>.agent.md
Instructions:    <scope>-<behavior>.instructions.md
Orchestrations:  <project>_<milestone>/
Rules:           <standard-name>.md or .mdc
Prompts:         <workflow-name>.prompt.md
Skills:          <skill-name>/SKILL.md
```

---

## 📊 Statistics

- **Agents**: 5 specialist agents
- **Instructions**: 3 scoped instructions
- **Rules**: 4 global rules
- **Prompts**: 3 workflow templates
- **Skills**: 28 reusable skills
- **Active Orchestrations**: 0
- **Archived**: 11 historical items

---

## 🔧 Maintenance

### Weekly
- [ ] Update active orchestration progress
- [ ] Check for completed work to archive

### Monthly
- [ ] Review archives for patterns to extract
- [ ] Update INDEX.md statistics
- [ ] Clean up temporary artifacts

### Quarterly
- [ ] Consolidate similar content
- [ ] Update subdirectory READMEs
- [ ] Review effectiveness and adjust

---

## 🆘 Help

### I can't find...
**Try**: `INDEX.md` → Quick Search section

### I don't know which to use...
**Agent vs Instruction**: [README.md](README.md#-quick-start) comparison  
**Prompt vs Agent**: [prompts/README.md](prompts/README.md#prompts-vs-agents-vs-instructions)

### I want to create...
**Check template**: Relevant subdirectory README → "Creating" section

### I have a question...
**Read**: Subdirectory README → FAQ or examples section  
**Ask**: Repository maintainers

---

## 🌟 Best Practices

### Do
✅ Read relevant READMEs before creating  
✅ Follow naming conventions  
✅ Update INDEX.md after changes  
✅ Archive completed work promptly  
✅ Extract patterns to skills  
✅ Document learnings

### Don't
❌ Create duplicates (check INDEX.md first)  
❌ Mix active and archived work  
❌ Skip documentation  
❌ Leave stale orchestrations  
❌ Hard-code project specifics  
❌ Ignore lifecycle guidelines

---

**Last Updated**: 2026-06-23  
**Version**: 1.1
