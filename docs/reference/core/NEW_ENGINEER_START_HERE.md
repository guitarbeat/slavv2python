# New engineer start here

[Up: Reference Docs](../README.md) · [Documentation Index](../../README.md) · [AGENTS.md § Work Decision Tree](../../../AGENTS.md#-work-decision-tree)

## In short

This repo has **two products**: a public pipeline (`slavv run`) and a MATLAB
translation audit (`slavv parity`). They share code but not the same defaults,
outputs, or definition of "done."

**Two meanings of "paper".** The **publication** is Mihelic et al. 2021 (PLOS
Comp Biol) — narrative only; the executable spec is
`external/Vectorization-Public/`. **Paper Path** is the public `paper` profile
(`slavv run`): Tracing Discovery, `float32`, C-order `[Z, Y, X]`. **Exact Route**
is the MATLAB-faithful cert path: Watershed Discovery, `float64`, F-order
`[Y, X, Z]`. There is no external paper for that split.
Citations: [papers/README.md](../papers/README.md).

---

## Pick your track

### Product engineer

Run the pipeline, add features, use Streamlit. No Experiment Root required.

```powershell
uv sync --extra app
uv run slavv run -i volume.tif -o slavv_output --export json
```

- [TUTORIAL.md](../../TUTORIAL.md) — first vascular extraction (Paper Path)
- [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) — pipeline stages and data flow
- [PYTHON_NAMING_GUIDE.md](../workflow/PYTHON_NAMING_GUIDE.md) — conventions
- [tests/README.md](../../../tests/README.md) — test placement

Skip for now: `workspace/`, ONE TRUTH, HANDOFF, parity CLI.

### Parity / exact-route engineer

Prove MATLAB parity, run the exact route, operate dests. Need Experiment Root
binaries on disk before `prove-exact` works.

```powershell
uv sync --extra app
uv run slavv parity inspect-experiment-root
```

- [PARITY_CERTIFICATION_GUIDE.md](../workflow/PARITY_CERTIFICATION_GUIDE.md) — how to run proofs (includes parity harness code tour)
- [ONE TRUTH](EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) — live pass/fail
- [HANDOFF.md](../../../.claude/HANDOFF.md) — operator commands and decision point
- [AGENTS.md § Domain Glossary](../../../AGENTS.md#domain-glossary) — canonical terms
- [Cheap Parity Ladder](../../../AGENTS.md#cheap-parity-ladder) — before any full-volume writer

Skip for now: investigations diary, ADR v5/v6 addenda.

---

## Jargon

| You hear… | You reasonably think… | Here it means… |
|-----------|----------------------|----------------|
| **Oracle** | Test mock, or Oracle™ | A **saved MATLAB run** on disk (`workspace/oracles/`) — the answer key |
| **Candidate** | Job applicant, or "maybe" | A **provisional edge** before cleanup — not "we're unsure about the project" |
| **Experiment** | Jupyter, or A/B test | Often **"compare two saved artifacts"** — not "run the pipeline experimentally" |
| **Dest** | Function argument / URL | A **named run folder** under `workspace/runs/.../` with a version suffix (`v18`, `v3`) |
| **Workspace** | IDE workspace, or `git workspace` | **Lab data directory** — like a dataset mount, not your VS Code window |
| **Exact** | "Precise enough" | **"Match MATLAB's lab implementation"** — including tie-break rules you cannot see in output JSON |

Full glossary: [AGENTS.md § Domain Glossary](../../../AGENTS.md#domain-glossary). Browsable mirror: [GLOSSARY.md](GLOSSARY.md).

---

## Reading this repo

There are many docs. Start with [docs/README.md](../../README.md) (the authority
map), not the investigations diary or ADR addenda. If `workspace/` looks empty on
a fresh clone, that is expected — Experiment Root binaries are not on GitHub.
Run `slavv parity inspect-experiment-root` after USB/rsync.

**"Closed" means Phase 1 exact-route certified on one full volume** — not
"backlog empty." Stretch (identical float bits) and Paper Path certification
are separate. TODO has open rows; that does not reopen Phase 1 unless
[ONE TRUTH](EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk)
says so.

---

## Common traps

### 1. Experiment Root and clone completeness

Git gives you code, docs, proof JSON. Git does **not** give you multi-GB
checkpoints, oracles, or datasets under [workspace/](../../../workspace/README.md).

First-week failure mode: run parity from docs → missing checkpoint errors →
assume the project is broken. USB/rsync Experiment Root binaries first, then
`slavv parity inspect-experiment-root`.

### 2. Run directory layout

Staged layout (`00_Refs/`, `01_Params/`, `02_Output/.../checkpoints/`,
`04_Edges/candidates.pkl`, `99_Metadata/`) is the real API between stages — not
the `slavv_output/` folder from the tutorial. Detail:
[workspace/README.md](../../../workspace/README.md).

Proof JSON pairing: a proof under one folder may describe another dest. Always
cite via `slavv parity inspect-proof`.

### 3. Code navigation

The repo has **two codebases behind two CLIs**:

| Entry | Role |
|-------|------|
| `slavv run` / `SlavvPipeline` | Pipeline writers — stage managers, `matlab_get_*.py` ports |
| `slavv parity` | Read/compare/launch orchestration under `analytics/parity/` |

Exact-route writers persist under a staged `run_dir`, not tutorial `slavv_output/`.

Parity harness orientation (four jobs, package map, writer vs reader, CLI
groups): [PARITY_CERTIFICATION_GUIDE.md § Parity Harness Code Tour](../workflow/PARITY_CERTIFICATION_GUIDE.md#parity-harness-code-tour).

### 4. MATLAB submodule

[external/Vectorization-Public](../../../external/Vectorization-Public) is canonical MATLAB source. Parity is proved against **executed oracles**, not transpilers ([Exact MATLAB Parity Rule](../../../AGENTS.md#exact-matlab-parity-rule)).

Do not diff production parity against uncommitted local `.m` edits.

### 5. Tooling footguns

- PowerShell-first; `uv run slavv` after `uv sync`
- `slavv jobs list` can hang; prefer `slavv monitor --once --run-dir …`
- Long exact Energy: joblib `Done N tasks` log leads `resume_state.json`
- Energy shape `(512,64,512)` on full volume = orientation bug, not float noise

---

## Maintainers

When updating this guide: run the [Docs Link Auditor](../../../.claude/agents/docs-link-auditor.agent.md) on links you touch; use [consolidate-concepts](../../../.claude/skills/consolidate-concepts/SKILL.md) if terminology drifts from AGENTS.md / GLOSSARY.md. Live status stays in [ONE TRUTH](EXACT_PROOF_FINDINGS.md) only — do not freeze KPIs or dest names here.
