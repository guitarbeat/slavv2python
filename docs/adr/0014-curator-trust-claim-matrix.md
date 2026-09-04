# ADR 0014: Curator Trust Claim Matrix

## In short

Only the **browser** Vertex→Edge→Apply curator may claim Trust-level MATLAB
familiarity. That claim is gated by a checkable matrix. Until every cell is
green, do not say “1:1 MATLAB” (or equivalent). Science Certification
(ADR 0011/0012) stays a separate Trust track.

## Status

Accepted (2026-09-04) — language gate in force; most evidence cells start red.

## Context

`STRATEGY.md` places the curation GUI under Trust (screens, keybindings, review
workflow). Three Python surfaces historically all spoke “MATLAB-style,” while
only the browser path encodes the documented ritual and `.slavv-curation.json`
session contract. Qt claimed “1:1” layout parity without a proof gate. Trust
language for the GUI was ahead of evidence.

## Decision

1. **Sole Trust claim surface:** the MATLAB-familiar browser curator
   (`slavv_python/interface/streamlit` matlab_curator / Vertex→Edge→Apply).
2. **Desktop Qt and napari** remain launchable as experimental / desktop review
   tools. They must not use Trust MATLAB-familiar or “1:1” claim language.
3. **Claim matrix** (below) is the authority for when Trust-certifying phrases
   are allowed. Designating the browser path as the Trust surface is allowed
   while cells are red; claiming proven MATLAB identity is not.
4. **Degraded browser modes** may exist but must not show Trust claim chrome.
5. **Science Certification** (prove-exact / ONE TRUTH / ADR 0011–0012) is not
   redefined here. Apply must not be gated on Network prove-exact.

### Claim matrix (claim surface = browser curator)

| Cell | Meaning | Initial status (2026-09-04) |
| --- | --- | --- |
| Claim-surface honesty | Only browser is labeled Trust MATLAB-familiar; desktop is not | **Green** after labeling pass in this change set |
| Layout-region honesty | Named regions / degraded disclosure match operator reality | **Red** — evidence deferred |
| Keybinding honesty | Documented keys match captured grammar | **Red** — evidence deferred |
| Vertex→Edge→Apply ritual | Two-stage Continue + Apply semantics match docs | **Red** until interaction evidence |
| Apply rebuild semantics | Apply rebuilds Network as documented; not science-gated | **Green** for “not science-gated”; full MATLAB compare deferred |
| Session Save/Load contract | `.slavv-curation.json` present on claim surface | **Green** for presence on browser; cross-host deferred |

**Language gate:** While any cell required for a Trust-certifying claim is red,
product and docs must not use “1:1 MATLAB,” “MATLAB-identical curator,” or
equivalent proven-identity phrasing for any Python curation UI.

## Consequences

- Operators see one named Trust path; desktop tools stay available without the
  Trust badge.
- Follow-on work (shared keymap, fidelity ladder, screenshot / interaction
  golden suite, MATLAB `.m` restore) supplies evidence to turn red cells green.
- Marketers and docs authors cite this ADR before asserting curator parity.

## Considered Options

| Option | Why rejected for this ADR |
| --- | --- |
| Multi-host Trust claims (browser + Qt + napari) | Inflates Trust language; hosts diverge on keys and ritual |
| Label cleanup only (no matrix) | No falsifiable bar; “1:1” can return quietly |
| Gate Apply on prove-exact Network green | Confuses human curation with science Certification |

## Related

- Plan: `docs/plans/2026-09-04-001-feat-curation-gui-trust-claim-surface-plan.md`
- Ideation: `docs/ideation/2026-09-04-matlab-familiar-curation-gui-ideation.html`
