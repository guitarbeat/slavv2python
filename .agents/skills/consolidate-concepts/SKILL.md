---
name: consolidate-concepts
description: >-
  Find duplicated or scattered definitions — both documentation concepts/domain
  terms and code-level definitions (constants, magic values/thresholds, helper
  functions, types/dataclasses/enums, config/parameter values) — and consolidate
  each into a single canonical home, updating references without changing meaning
  (docs) or behavior (code). Use when deduplicating docs or code, merging
  scattered or duplicated definitions or terminology, deduplicating repeated
  constants/thresholds/helpers/types, establishing a single source of truth for
  concepts, values, or symbols, extracting a shared definition, or reconciling
  documentation or code drift.
---

# Consolidate Concepts

Consolidate duplicated and scattered definitions — **documentation** concepts
and domain terms, and **code** definitions (constants, thresholds, helpers,
types, config values) — into one canonical home, replace or delete the
duplicates, and update references. Do this **without changing meaning** (docs)
or **without changing behavior** (code).

Two parallel tracks run through every phase:

- **Docs track** — concept/term definitions and explanatory prose in Markdown.
- **Code track** — duplicated symbols/values in source: constants, magic
  numbers/thresholds, copy-pasted helper functions, parallel type/dataclass/enum
  definitions, and config/parameter values redefined in several places.

Apply whichever track(s) the task calls for; the phases are shared.

## When to Use

**Docs track:**

- The same term or concept is defined in multiple docs, with wordings that have drifted.
- Explanations of one concept are scattered across several files.
- The user wants a single source of truth (glossary or designated doc) for domain terms.
- Documentation drift: two docs describe the same thing differently and should agree.

**Code track:**

- The same constant, threshold, or magic value is defined in several modules.
- A helper function is copy-pasted (verbatim or near-verbatim) across files.
- Parallel type/dataclass/enum definitions describe the same shape in multiple places.
- Config or parameter values (defaults, limits, keys) are duplicated across modules.
- A value or symbol should have a single source of truth imported everywhere.
- Code drift: two copies of the same definition have diverged and should be reconciled.

## Guiding Principle

- **Preserve meaning (docs).** Consolidation is editorial reorganization, not a rewrite. The union of information must survive.
- **Preserve behavior (code).** Consolidation is a refactor (extract-and-reference), not a rewrite. Runtime behavior must be identical: prefer a single canonical definition imported everywhere, and change no observable value or logic.
- **One canonical definition per concept, value, or symbol.**
- **Never merge coincidental equals.** Two constants/symbols that happen to have the same value but mean different things are distinct — do not merge them. The same applies to two terms that merely share a name.
- **Leave a pointer.** Where a duplicate doc is removed, leave a short link to the canonical home. Where a duplicate symbol is removed, replace it with an import/reference to the canonical definition, not a silent gap.
- **Merge nuance, don't truncate.** When doc variants differ in detail, combine their detail into the canonical definition rather than keeping only the shortest one.
- **Never silently pick a winner.** When two definitions genuinely conflict — doc definitions with incompatible meaning, or code definitions with different values/logic — stop and surface the conflict to the user; do not merge or discard either.

Copy this checklist and track progress:

```
Consolidation Progress:
- [ ] Phase 1: Inventory concepts, values, and symbols and their locations
- [ ] Phase 2: Choose canonical home; list conflicts/drift
- [ ] Phase 3: Plan and get user confirmation
- [ ] Phase 4: Execute merges, replacements, reference/import updates
- [ ] Phase 5: Verify against checklist
```

## Phase 1 — Inventory

Respect the repo's search-exclusion norms in **both** scans: skip vendored /
third-party trees, caches, generated output, build artifacts, and any paths the
repo marks excluded.

**Docs scan.** Scan the repository's Markdown docs. For every concept or domain
term, build a `term -> {locations, variants, references}` map recording:

- **Locations** — every file plus heading (or line) where the term is *defined or explained*.
- **Variants** — the distinct wordings used at each location.
- **Agreement** — whether the variants agree, drift (same intent, different detail/phrasing), or conflict (incompatible meaning).
- **References** — every doc that *mentions or links to* the term (inbound references and anchors).

Focus on definitions and explanatory sections, not every incidental mention. A
term used identically in many places is fine; the targets are terms *defined*
in more than one place.

**Code scan.** Scan the source tree in parallel. For every duplicated value or
symbol (constant, threshold/magic value, helper function, type/dataclass/enum,
config/parameter value), build a `symbol -> {definitions, variants, references}`
map recording:

- **Definitions** — every definition site: file plus symbol name (and line). For a magic value, the literal and each place it is hard-coded.
- **Variants** — the distinct implementations/values at each site.
- **Agreement** — whether the definitions **agree** (identical), **drift** (same intent, minor differences — e.g. a helper with cosmetic edits, a constant with different formatting), or **conflict** (different value or different logic/behavior).
- **References** — every call site, import, and usage that depends on each definition.

Target genuinely shared definitions, not every coincidental literal (a `0`, `1`,
or `""` used locally is not a concept). A value repeated across modules that
*means the same thing* is a target; the same numeric literal used for unrelated
purposes is **not**.

## Phase 2 — Choose the Canonical Home

**Docs track:**

1. **Prefer an existing source of truth.** If the repo already has a glossary or designated canonical doc (e.g. an `AGENTS.md` glossary section, a `GLOSSARY.md`, or an architecture reference), use it.
2. **Otherwise propose creating one.** Suggest a single home (e.g. a new `GLOSSARY.md` or a "Concepts" section in an existing top-level doc) and let the user confirm before creating it.

**Code track:**

1. **Prefer an existing shared module.** If the repo already has a natural home — a `constants`/`config` module, a `utils/` package, or a shared `types` module — put the canonical definition there.
2. **Otherwise propose creating one.** Suggest a single shared module and let the user confirm before creating it.
3. **Respect boundaries.** Choose a home that respects the repo's package/layer boundaries and import direction (place shared definitions low enough in the dependency graph that all consumers can import them without creating cycles). Note and honor the repo's file-size limits, if any, when adding to or creating a module.

**Both tracks — list conflicts and drift.** From the Phase 1 maps, produce an
explicit list of every concept/value/symbol whose variants drift or conflict,
with the differing text/values/implementations side by side. Mark which are
drift (mergeable) vs conflict (must escalate).

### Canonical vs. mirrored view

Some repos maintain both a **canonical** definition (the source of truth) and a
**mirrored/browsable view** of the same terms (e.g. a table-form glossary).
Example pattern: a canonical `AGENTS.md § Domain Glossary` that is auto-loaded
into agent context, plus a `GLOSSARY.md` that mirrors it for browsing and adds
technical detail — governed by two rules: *if the two disagree, the canonical
section wins*, and *when adding or redefining a term, update both*. If the repo
uses this pattern, treat the canonical location as the home and keep the
mirrored view in sync (do not drop it).

## Phase 3 — Plan & Confirm

Present a plan and get **explicit user confirmation before editing**. Include:

- **Items to centralize** — which doc terms and which code values/symbols move to the canonical home, with their source locations.
- **Canonical home** — the chosen (or proposed) source-of-truth path/module for each, plus any mirrored view to keep in sync.
- **Items to keep local** — terms used in only one doc, and single-use constants/helpers/types that should *not* be centralized.
- **Per-duplicate handling:**
  - *Docs:*
    - **replace-with-link** — swap the duplicate definition for a one-line pointer to the canonical home.
    - **delete** — remove a fully redundant restatement (still leave a link if anything referenced it).
    - **merge-then-link** — fold the variant's extra nuance into the canonical definition, then replace the local copy with a pointer.
  - *Code:*
    - **replace-with-import** — delete the local duplicate and import/reference the canonical definition.
    - **delete** — remove a redundant re-definition, repointing usages to the canonical symbol.
    - **merge-then-import** — reconcile a drifting variant into one canonical definition, then import it everywhere (only for true drift, never for conflicts).
- **Conflicts** — the escalation list from Phase 2, asking the user to resolve genuine conflicts (incompatible doc meaning, or differing code values/logic) before execution.

> **Pitfall — keep it local:** Do not centralize a term that appears in only one
> doc, or a constant/helper/type used in only one place. Over-centralizing adds
> indirection and coupling for no benefit. Centralize only shared, cross-file
> concepts, values, and symbols.

## Phase 4 — Execute

Only after confirmation.

**Docs track:**

1. **Write the merged canonical definition** into the source-of-truth, reconciling drifting variants so no nuance is lost. For any genuine conflict, use the resolution the user chose (never silently merge).
2. **Handle each duplicate** per the confirmed plan: replace with a concise reference/link to the canonical home, or remove it if fully redundant. A replaced section should read like "See [Term](path#anchor)." rather than vanishing.
3. **Update inbound references.** Repoint every link/anchor that targeted a removed or renamed section so nothing dangles. Update heading anchors that changed.
4. **Sync any mirrored view.** If the repo maintains a browsable mirror of the canonical glossary, update it too, and preserve the "canonical wins + keep the view in sync" rules.

**Code track:**

1. **Write the canonical definition** into the chosen shared home. For true drift, reconcile variants into one definition that preserves the intended behavior. For any genuine conflict, use the user's resolution (never silently merge differing values/logic).
2. **Replace each duplicate** with an import/reference to the canonical definition; delete the local copy. Update every call site, import, and usage to point at the canonical symbol.
3. **Preserve the public API.** If a moved symbol was part of a module's public surface, re-export it from the old location (or otherwise keep the import path working) so external callers don't break.
4. **Keep behavior identical**, then validate with the repo's own tooling — run its tests, type-checker, and linter/formatter (use the repo's documented commands) and confirm they pass before reporting done.

## Phase 5 — Verify

Confirm every item before reporting done.

**Docs track:**

- [ ] Every consolidated concept has **exactly one** canonical definition.
- [ ] **No duplicate definitions** remain anywhere in the docs.
- [ ] Every removed/replaced section leaves a **working link** to the canonical home.
- [ ] **All inbound references and anchors still resolve** (no broken links).
- [ ] **No meaning or nuance lost** — diff the merged canonical text against the *union* of the original variants and confirm all distinct detail survived.

**Code track:**

- [ ] Every consolidated value/symbol has **exactly one** canonical definition.
- [ ] **No duplicate definitions** remain (no copy-pasted helpers, re-declared constants, or parallel types).
- [ ] **Every reference/import resolves** — call sites, imports, and re-exports all point at the canonical definition.
- [ ] **Behavior is unchanged** — the repo's tests, type-checker, and linter all pass.
- [ ] **No coincidentally-equal-but-distinct values were merged** — each consolidation joined only semantically identical definitions.

**Both tracks:**

- [ ] **Genuine conflicts were escalated** to the user and resolved explicitly, not silently merged or dropped.
- [ ] Any **mirrored/browsable view is in sync** with the canonical source.
- [ ] Items correctly **kept local** were not centralized.

## Common Pitfalls

**Docs track:**

- **Over-centralizing local terms.** Single-use or doc-internal terms belong where they are used. Centralizing them adds indirection without payoff.
- **Losing nuance.** Picking the shortest variant and discarding another variant's extra detail changes meaning. Merge the detail instead.
- **Broken anchors.** Removing or renaming a heading orphans every link pointing at its anchor. Update inbound references whenever an anchor changes.
- **Conflating distinct concepts.** Two things sharing a name (or a near-identical name) are not necessarily the same concept. Confirm they mean the same thing before merging.
- **Forgetting the mirror.** When a repo keeps both a canonical glossary and a browsable view, updating only one reintroduces the exact drift you are trying to remove.

**Code track:**

- **Merging semantically-distinct-but-equal values.** Two constants that share a value today but model different things will drift apart tomorrow. Never merge on value alone — merge only on shared meaning.
- **Introducing circular imports.** Choosing a canonical home that sits above its consumers in the dependency graph creates cycles. Place shared definitions low enough that everyone can import them.
- **Breaking the public API.** Moving a symbol without re-exporting it from its old location breaks external callers. Preserve import paths (re-export) when a symbol is public.
- **Violating boundaries or size limits.** Don't cross package/layer boundaries the repo enforces, and don't blow past its file-size limits when adding to or creating a shared module.

**Both tracks:**

- **Silently resolving conflicts.** If two definitions truly disagree — incompatible doc meaning, or different code values/logic — do not choose for the user; surface both and ask.

## Validation Checklist

- [ ] User confirmed the consolidation plan before any edits.
- [ ] Each consolidated concept/value/symbol has exactly one canonical definition in the chosen source of truth or shared module.
- [ ] No duplicate definitions remain; removed doc duplicates leave a working pointer, and removed code duplicates import/reference the canonical definition.
- [ ] All inbound doc references and heading anchors resolve; all code references, imports, and re-exports resolve.
- [ ] Merged canonical docs preserve the union of all variants' meaning and nuance (verified by diff).
- [ ] Code behavior is unchanged: the repo's tests, type-checker, and linter pass.
- [ ] No coincidentally-equal-but-semantically-distinct values or symbols were merged.
- [ ] Genuine conflicts were escalated and resolved by the user, not silently merged.
- [ ] Any mirrored/browsable glossary view was updated to match, preserving the "canonical wins + keep in sync" rule.
- [ ] Terms and symbols that should stay local remained local.
