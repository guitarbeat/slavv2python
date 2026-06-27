---
name: self-improving-agent
description: Capture durable corrections, repeated workflow lessons, and agent-process improvements for slavv2python. Use when the user explicitly asks the agent to learn, remember, improve itself, adapt an external self-improvement skill, reflect on completed work, mine repeated corrections, or promote a reusable project workflow into AGENTS.md, docs, solutions, TODO, or .claude/skills.
---

# Self Improving Agent

Improve future agent behavior by turning real evidence into small durable instructions. Do not create memory churn; capture only lessons that should help another agent on this repository.

## What To Capture

Capture a lesson when at least one is true:

- The user gives an explicit durable preference: "always", "never", "remember", "I prefer", or "stop doing".
- The same correction or workflow problem appears repeatedly.
- A debugging, parity, CI, or review workflow produced a reusable procedure.
- A project-specific rule belongs in an existing skill, `AGENTS.md`, `docs/solutions`, `docs/reference/core/EXACT_PROOF_FINDINGS.md`, or `docs/TODO.md`.
- A completed multi-step task exposed an avoidable failure mode for future agents.

Do not capture:

- One-off task instructions.
- Private credentials, local secrets, or transient file paths unless they are documented repo paths.
- Unverified guesses or vibes.
- Long chat summaries.
- Lessons that duplicate `AGENTS.md`.

## Reflection Workflow

1. Identify the trigger: correction, repeated pattern, successful workflow, failure, or explicit request.
2. Gather evidence from the current task: commands, tests, docs consulted, user feedback, and final outcome.
3. Decide the storage target.
4. Write the smallest useful entry.
5. Validate that the new instruction does not conflict with `AGENTS.md`.
6. Mention the captured lesson in the final response only if a file was changed.

Use this reflection prompt after substantial work:

```text
Did this task reveal a durable project rule, reusable workflow, or correction?
If yes, where should it live so the next agent will actually find it?
If no, do not write memory.
```

## Storage Targets

- `.claude/skills/<skill>/SKILL.md`: recurring multi-step agent behavior with clear triggers.
- `docs/solutions/<slug>.md`: a concrete fix, parity workflow, or integration resolution that future debugging should find.
- `docs/reference/core/EXACT_PROOF_FINDINGS.md`: live exact-parity run status, blockers, champion baselines, and parity solution index.
- `docs/TODO.md`: active tasks and planning hub entries, not detailed run logs.
- `AGENTS.md`: broad repository rules only after explicit user approval.
- `workspace/scratch/agent-reflections.md`: temporary reflections that are not yet strong enough for docs or skills.

Prefer improving an existing skill or doc over creating a new one. Create a new skill only when the trigger and workflow are distinct.

## Entry Formats

For `workspace/scratch/agent-reflections.md`:

```markdown
## YYYY-MM-DD - <short topic>

- Trigger: <correction, failure, repeated pattern, or explicit request>
- Evidence: <brief local evidence>
- Lesson: <durable instruction>
- Candidate home: <skill, doc, solution, TODO, or AGENTS.md>
- Confidence: strong | medium | weak
```

For `docs/solutions/<slug>.md`, use YAML frontmatter compatible with this repo's documented solutions index:

```markdown
---
title: <short title>
module: <repo area>
tags: [<tag>, <tag>]
problem_type: <bug | parity | workflow | integration | ci>
---

# <Title>

## Problem
## Evidence
## Solution
## Verification
```

## Promotion Rules

- Promote to a skill when the lesson is a reusable agent workflow with a clear trigger.
- Promote to docs when it clarifies project architecture, parity practice, or run-state behavior.
- Promote to a solution when it records a reproducible problem and verified fix.
- Keep in scratch when useful but unproven.
- Delete or ignore scratch notes when later evidence contradicts them.

When a lesson concerns MATLAB parity, read `docs/reference/core/EXACT_PROOF_FINDINGS.md` first and follow the exact parity rule: no undocumented approximations.

For parity proof sessions, always decide after the first `prove-exact` pass or failure whether the result belongs in `EXACT_PROOF_FINDINGS.md`, a `solution-note-generator` note, a skill edit, or scratch. Do this before moving on to downstream stages so the evidence stays close to the command that produced it.

## Generative Use With Find Agent Skills

When asked to use skills generatively:

1. Use `find-agent-skills` to inventory local skills and run `suggest_skill_candidates.py`.
2. Treat each candidate as a hypothesis, not an instruction to create files automatically.
3. Check whether the evidence is verified, repeated, and project-specific.
4. Promote in this order:
   - Edit an existing skill when a trigger or workflow needs sharpening.
   - Use `solution-note-generator` when the evidence is a verified fix.
   - Add a new skill only when no existing skill owns the workflow.
   - Add a scratch reflection when the signal is promising but weak.
5. After promotion, run the skill validator for any changed skill.

This creates a bounded improvement loop:

```text
repo signals -> candidate scanner -> human/agent judgment -> skill/doc/solution promotion -> validation
```

## Safety

Do not silently rewrite repository-wide rules. If a lesson would change `AGENTS.md`, ask or clearly state the proposed change first unless the user explicitly requested that edit.

Never let self-improvement become a second task that delays the user's primary request. Capture small, high-signal lessons after the work is done or when the user specifically asks for it.
