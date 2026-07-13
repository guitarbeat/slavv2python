# Experiment analysis template

[Up: Reference Docs](../README.md)

Use this template for maintained experiment-analysis documents when a current
engineering question needs hypothesis, methodology, interpretation, and next
steps. Do not use it for live run status; live exact-parity status belongs in
[EXACT_PROOF_FINDINGS.md](../core/EXACT_PROOF_FINDINGS.md).

```markdown
# <Experiment name>

[Up: Reference Docs](../README.md) · [Live status](../core/EXACT_PROOF_FINDINGS.md)

This document is a maintained planning aid, not the live status log.

## Experiment question

What question does this experiment answer?

## Hypothesis

What is the current best explanation, and what observation would falsify it?

## Methodology

What surfaces, commands, datasets, or probes are used? Link to operator docs for
commands instead of duplicating unstable command blocks.

## Current results

Summarize only stable baseline numbers needed to orient the reader. Point to the
live status source for current results.

| Surface | Metric | Current value | Interpretation |
|---|---:|---:|---|
| <surface> | <metric> | <value> | <meaning> |

## Interpretation

What does the evidence imply? Which hypotheses are deprioritized?

## Limitations

What does this experiment not prove?

## Next steps

1. <Next measurement or implementation step>
2. <Next verification step>

## Done criteria

What exact evidence closes the experiment?

## Source ownership

- Live status:
- Operator commands:
- Tasks:
- Figures / reporting:
```

## Source ownership rules

- **Live status:** `EXACT_PROOF_FINDINGS.md`
- **Operator commands:** `.claude/HANDOFF.md` or a workflow guide
- **Tasks:** `docs/TODO.md`
- **Requirements:** `docs/plans/*-spec.md`
- **Publication / figure summaries:** `figures/README.md` and generators
