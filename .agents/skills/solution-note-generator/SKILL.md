---
name: solution-note-generator
description: Promote verified fixes, parity discoveries, integration resolutions, and reusable runbooks into searchable docs/solutions notes. Use after a non-trivial problem is solved and verified, when docs mention /ce-compound, when self-improving-agent chooses a solution-note target, or when a future agent needs a durable record of evidence, solution, and verification without adding status noise to TODO or EXACT_PROOF_FINDINGS.
---

# Solution Note Generator

Create a concise, searchable solution note only after the problem is solved and verified. This skill turns hard-won debugging into durable project memory.

## Preconditions

Before writing a note, verify all are true:

- The problem is non-trivial or likely to recur.
- The root cause or workflow is understood.
- The fix or runbook has verification evidence.
- The note will not duplicate an existing `docs/solutions` entry.

Do not write solution notes for in-progress debugging, unverified hypotheses, active run status, or one-off task context.

## Destination

Choose the folder by problem shape:

- `docs/solutions/integration-issues/`: oracle promotion, file formats, dependency boundaries, environment compatibility.
- `docs/solutions/best-practices/`: reusable workflow or design practice.
- `docs/solutions/parity/`: parity-specific proof failures and fixes when the details are more than a one-line findings index.
- Create a new category only when the existing categories do not fit.

For parity-related notes, add a one-line index row to `docs/reference/core/EXACT_PROOF_FINDINGS.md` under "Compound learnings"; do not copy the full solution there.

## Frontmatter

Use YAML frontmatter so agents can search and classify notes:

```markdown
---
title: <short title>
module: <repo area, e.g. analytics/parity or processing/energy>
tags: [<tag>, <tag>]
problem_type: <bug | parity | workflow | integration | ci>
resolution_type: <code_fix | docs_fix | runbook | diagnosis>
---
```

## Body

Use this shape:

```markdown
# <Title>

## Problem
<What failed, where it surfaced, and why it mattered.>

## Evidence
<Commands, proof output, first failing artifact, stack trace, or local diagnostic.>

## Root Cause
<The smallest accurate explanation.>

## Solution
<The fix or runbook future agents should reuse.>

## Verification
<Exact command(s), result, and remaining caveats.>

## Follow-Up
<Only durable next steps. Omit if none.>
```

## Workflow

1. Search existing notes:

```powershell
rg -n "<keyword>|<module>|<error>" docs\solutions docs\reference\core\EXACT_PROOF_FINDINGS.md
```

2. Gather proof: failing command, fixed command, before/after result, or diagnostic file.
3. Pick a slug in lowercase hyphen-case.
4. Write the note with explicit `encoding="utf-8"` if using a script, or edit with normal repo tooling.
5. If parity-related, add a one-line index in `EXACT_PROOF_FINDINGS.md`.
6. Run a docs sanity search for the slug and tags.

## Quality Bar

- Prefer exact command lines over vague summaries.
- Name the affected Stage Result, Checkpoint, Oracle, Parity Run, or Artifact when relevant.
- Keep active status in `EXACT_PROOF_FINDINGS.md`, not in the solution body.
- Keep tasks in `docs/TODO.md`, not in the solution body.
- Keep the note short enough that a future agent can read it before making a fix.
