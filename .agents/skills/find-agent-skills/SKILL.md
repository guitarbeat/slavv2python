---
name: find-agent-skills
description: Discover and evaluate reusable agent skills for a requested task. Use when the user asks to find, install, adapt, compare, or recommend skills; asks whether an agent can gain a capability; mentions npx skills, OpenClaw skills, Claude/Codex skills, or a skill marketplace; or when a task would benefit from checking this repository's existing .agents/skills before creating new workflow guidance.
---

# Find Agent Skills

Find the smallest trustworthy skill surface that helps with the user's task. Prefer project-local skills and documented workflows before suggesting external installs.

## Search Order

1. Search this repository's `.agents/skills` first.
2. Search Codex-discoverable tools with `tool_search` when the user asks for a connector, plugin, thread tool, automation, or MCP-backed capability.
3. Search global/user skills only if local skills do not cover the task.
4. Search external registries or GitHub only when the user explicitly wants new skills or there is no local fit.
5. Adapt external skill ideas into `.agents/skills` when the user wants project-specific behavior.

Use `scripts/find_local_skills.py` for a quick local inventory:

```powershell
python .agents\skills\find-agent-skills\scripts\find_local_skills.py parity
python .agents\skills\find-agent-skills\scripts\find_local_skills.py "PR review"
```

Use `scripts/suggest_skill_candidates.py` when the task is generative: audit active repo work and propose the next skill, skill edit, or solution note.

```powershell
python .agents\skills\find-agent-skills\scripts\suggest_skill_candidates.py
python .agents\skills\find-agent-skills\scripts\suggest_skill_candidates.py --format markdown
```

## Evaluation

For each candidate skill, verify:

- Trigger fit: the description matches the actual user request.
- Scope fit: the skill works in `slavv2python` without fighting `AGENTS.md`.
- Trust: local project skills are safest; known official sources are next; unknown packages require source review before install.
- Blast radius: avoid skills that run remote code, alter credentials, or write outside the repo unless the user explicitly approves.
- Freshness: for external claims, check current source docs or repository content before recommending.

Do not recommend a skill based only on a marketplace title, install count, or SEO page. Inspect the `SKILL.md` or source repository enough to know what it will ask future agents to do.

## Installing Or Adapting

- For Codex-local project behavior, create or update `.agents/skills/<skill-name>/SKILL.md`.
- For global Codex behavior, use the user's global skills directory only when they explicitly ask for a machine-wide install.
- For app/plugin capabilities, use `tool_search` first, then app/plugin install tools only if the requested tool is unavailable and explicitly requested.
- For external skill packages, prefer reading the source and adapting the instructions over blind installation.

When adapting external skills for this repo:

1. Translate tool-specific commands into Codex-compatible workflows.
2. Replace generic memory paths with repo-owned docs or `workspace/scratch`.
3. Add `slavv2python` domain references: parity, oracles, stage results, run state, and testing guide.
4. Keep the skill concise; move long examples into references only when needed.
5. Validate with the skill creator validator.

## Generative Skill Loop

Use this loop when the user asks how to make skills compound, asks for a codebase audit, or asks what skills to create next:

1. Run the local inventory and candidate scanner.
2. Compare active tasks in `docs/TODO.md` and live blockers in `docs/reference/core/EXACT_PROOF_FINDINGS.md` against existing skill descriptions.
3. Classify each gap as:
   - skill edit: an existing skill needs a sharper trigger or workflow.
   - new skill: a repeated workflow has no clear trigger surface.
   - solution note: a verified fix should be searchable, but not a recurring workflow.
   - scratch reflection: useful but not yet proven.
4. Prefer skill edits over new skills unless the trigger is distinct.
5. Hand promotion decisions to `self-improving-agent` so durable lessons land in the correct repo surface.

## Response Shape

When the user asks for options, return:

```text
Best fit: <skill or "none">
Why: <one paragraph>
Use it by: <trigger phrase or command>
Risks: <trust or scope concerns>
Next step: <install/adapt/use directly>
```

When no skill fits, say so and solve the task directly with the repo's normal workflow.
