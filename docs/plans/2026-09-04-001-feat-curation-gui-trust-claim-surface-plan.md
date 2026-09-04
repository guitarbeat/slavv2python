---
title: "Plan: Curation GUI Trust claim surface"
type: feat
date: 2026-09-04
topic: curation-gui-trust-claim-surface
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-brainstorm
execution: code
upstream_ideation: docs/ideation/2026-09-04-matlab-familiar-curation-gui-ideation.html
---

# Plan: Curation GUI Trust claim surface

## Goal Capsule

- **Objective:** Make Trust language for the curation GUI falsifiable: only the MATLAB-style browser curator may claim MATLAB familiarity at Trust level, and that claim is gated by a checkable claim-matrix ADR.
- **Product authority:** This plan owns claim-surface policy, operator-facing labeling, and the claim-matrix ADR. Layout fidelity work, shared keymaps, session interchange on desktop hosts, screenshot golden suites, and MATLAB `.m` source restoration are **not** active scope.
- **Open blockers:** None.

---

## Product Contract

### Summary

Establish a single Trust claim surface for MATLAB-familiar curation (the browser curator) and an ADR claim matrix that bans “1:1 MATLAB” / Trust-certifying phrases until listed cells are green. Qt and napari remain available as experimental or desktop tools with honest labels.

### Problem Frame

`STRATEGY.md` places curation under Trust (screens, keybindings, review workflow), but three Python surfaces all speak “MATLAB-style” while only the browser path encodes the documented Vertex→Edge→Apply ritual and session contract. Qt already claims layout parity without a proof gate. Without a claim surface and matrix, Trust language for the GUI stays ahead of evidence the same way science Trust would if every dest could call itself certified.

### Key Decisions

- KD1. **Browser is the sole Trust MATLAB-familiar claim surface.** (session-settled: user-approved — chosen over multi-host Trust claims: ideation top pick deferred to agent judgment.) Governs R1, R2, R6.
- KD2. **Claim-matrix ADR is the enforcement mechanism.** (session-settled: user-approved — chosen over label-only cleanup: same deference.) Governs R3, R4, R5, R7.
- KD3. **Policy before interaction parity builds.** This plan ships governance and labeling; keymap, 2×2 lock, golden suite, and `.m` restore stay follow-ons. Governs Scope Boundaries.
- KD4. **Degraded browser modes may exist but do not inherit the Trust badge.** Fallback UX is allowed; Trust MATLAB-familiar labeling is not. Governs R3, R5.

<!-- ce-section: work-relationships -->
### How This Work Fits Together

This plan owns **Trust claim governance** for the curation GUI only. Surrounding areas below are the current understanding, not a committed roadmap.

- Shared interaction grammar / keymap-as-data
  - **Depends on** a named claim surface (this plan) so the grammar has one Trust host to target first
  - **Can proceed independently of** ADR cell greens once the claim surface is named
- Fidelity ladder (grammar → regions → optional classical 2×2)
  - **Shares** Trust language rules with this plan’s matrix cells
  - **Outside** this plan’s DoD
- Session JSON interchange on Qt/napari; screenshot / interaction-transcript golden suite; pin/restore MATLAB curator `.m` sources
  - **May be informed by** matrix cells this ADR defines
  - **Outside** this plan’s identity
- Exact-route science Certification (prove-exact / ONE TRUTH)
  - **Independent** of curator claim governance; Apply must not be gated on Network prove-exact

### Actors

- A1. Maintainer / Trust steward — owns STRATEGY/Trust wording and ADR claim status.
- A2. Operator / curator — chooses a curation workflow and must see honest labels.
- A3. Docs / UI author — updates workflow copy and UI strings to match the claim surface.
- A4. Implementation agent — delivers ADR text, string diffs, degraded-mode chrome, and tests without inventing new claim surfaces.

### Requirements

**Claim surface**

- R1. Exactly one surface may claim Trust-level MATLAB familiarity: the MATLAB-style browser curator (Vertex→Edge→Apply path already documented as the default browser curator).
- R2. Desktop Qt and experimental napari workflows remain launchable when dependencies allow, but operator-facing labels and docs must not describe them as Trust-level MATLAB-familiar or “1:1 MATLAB.”
- R3. Any browser degraded or fallback presentation (for example single-view / projection-limited mode) must not display the Trust MATLAB-familiar badge or equivalent claim language while degraded.

**Claim matrix**

- R4. An ADR defines a curator Trust claim matrix with checkable cells covering at least: claim-surface honesty, layout-region honesty (including degraded-mode disclosure), keybinding honesty, Vertex→Edge→Apply ritual, Apply rebuild semantics, and session Save/Load contract presence on the claim surface.
- R5. Until every matrix cell for the claim surface is marked green under the ADR’s own criteria, product and docs must not use “1:1 MATLAB,” “MATLAB-identical curator,” or equivalent Trust-certifying phrases for any Python curation UI. Designating the browser path as the Trust claim surface (per R1/A4) is allowed while cells are red; claiming proven MATLAB identity is not.
- R6. Workflow and Curation page copy must name the browser path as the Trust MATLAB-familiar surface and describe Qt/napari as desktop/experimental review tools without Trust MATLAB-familiar claims.
- R7. Science Certification language (prove-exact, ONE TRUTH, ADR 0011/0012) remains distinct from curator Trust claims; this work must not redefine Phase 1 closure or gate Apply on Network prove-exact.

### Key Flows

- F1. **Operator picks a curation workflow** — Sees browser option labeled as the Trust / MATLAB-familiar path; sees desktop options labeled without Trust MATLAB-familiar claims. Covers R1, R2, R6.
- F2. **Trust steward updates claim status** — Marks matrix cells green/red per ADR criteria; when any cell is red, marketing and UI claim strings stay suppressed per R5. Covers R4, R5.
- F3. **Operator hits degraded browser mode** — Can still curate with disclosed limits; Trust MATLAB-familiar badge is absent. Covers R3.

### Acceptance Examples

- AE1. **Happy path labels** — Given Curation workflow chooser is shown, when the operator reads the options, then only the browser curator carries Trust MATLAB-familiar wording and desktop options do not. Covers R1, R2, R6.
- AE2. **Red matrix blocks 1:1 copy** — Given any claim-matrix cell is red, when docs or UI are reviewed for claim language, then no “1:1 MATLAB” (or equivalent) string appears for Python curation UIs. Covers R5.
- AE3. **Degraded mode** — Given the browser curator is in a documented degraded presentation, when the operator views chrome/status, then Trust MATLAB-familiar claim language is not shown. Covers R3.
- AE4. **Apply still human** — Given Edges exist on a non-certified run, when the operator reaches Apply, then Apply remains available without requiring Network prove-exact green. Covers R7.

### Edge Cases

- E1. Historical docs or code comments that already say “1:1” on Qt — must be corrected or demoted as part of this work’s labeling pass (per R2, R5), not left as silent exceptions.
- E2. Headless / no-display environments — desktop options stay disabled or unavailable as today; claim rules still apply to any remaining visible copy.

### Scope Boundaries

**In scope**

- Naming the browser curator as the sole Trust MATLAB-familiar claim surface
- Claim-matrix ADR and red/green claim language rules
- Operator-facing labels and maintained workflow/docs copy for curation entry points
- Demoting over-claiming strings on Qt/napari paths
- Degraded-mode Trust badge suppression in the browser curator chrome

**Out of scope**

- Implementing a shared keymap / interaction-grammar module
- Classical four-panel 2×2 layout lock or fidelity-ladder UI builds
- Porting `.slavv-curation.json` session interchange to Qt/napari
- Screenshot or interaction-transcript golden suite / CI harness
- Restoring or pinning missing MATLAB curator `.m` sources
- Changing Apply semantics, Network rebuild behavior, or science Certification gates
- Automatic/ML curation workflows (labeling only if they currently misuse Trust MATLAB-familiar claims)

### Assumptions

- A1. The existing browser Vertex→Edge→Apply curator is the correct claim-surface candidate (already the documented default).
- A2. Degraded browser modes will continue to exist for some volumes; the product response is badge suppression, not removal of the fallback.
- A3. A later plan will flesh matrix cell evidence (screenshots, key transcripts, `.m` compare); this plan only requires the matrix definition and language gate.
- A4. Naming the designated Trust surface (“Trust path: browser curator”) is allowed while matrix cells remain red; “1:1” / “MATLAB-identical” / certified-parity phrasing is not (per R5).

### Success Criteria

- S1. A reader of STRATEGY Trust + Curation entry UI can identify exactly one Trust MATLAB-familiar surface.
- S2. A claim-matrix ADR exists and is the cited authority for when Trust MATLAB-familiar / 1:1 language is allowed.
- S3. Qt/napari and degraded browser modes do not carry Trust MATLAB-familiar claim language while matrix cells required for that claim are red or while degraded.

---

## Planning Contract

### Key Technical Decisions

- KTD1. **ADR number is 0014; matrix lives as a markdown table in the ADR.** (session-settled: user-approved — planning deferred Q1 resolved by agent: checklist in ADR over a machine-readable status file for v1.) Initial cell statuses after this change set: `claim-surface honesty` may go green when U2–U4 land; all other R4 cells stay red until follow-on evidence plans. Governs U1.
- KTD2. **Forbidden vs allowed claim language.** Forbidden while any matrix cell is red: `1:1`, `MATLAB-identical`, `feature parity with the MATLAB GCI` as a Trust claim. Allowed: naming the browser path as the designated Trust / MATLAB-familiar surface. Desktop labels use “desktop review” / “experimental” without “MATLAB-style.” Governs U2–U5. Covers R5, R2.
- KTD3. **STRATEGY gets a one-line ADR pointer** under Trust / equivalence. (Resolves deferred Q3: yes, same change set.) Governs U4.
- KTD4. **Degraded-mode Trust chrome** — when `degradedReason` is set, the browser curator must not show Trust MATLAB-familiar badge/title chrome; keep the existing degraded reason disclosure. Governs U5. Covers R3.
- KTD5. **String inventory for demotion** (resolves Q2): `slavv_python/interface/streamlit/views/curation.py` radio/help; `slavv_python/visualization/interactive_curator.py` module docstring; `docs/reference/workflow/MANUAL_CURATION_WORKFLOW.md`; `docs/reference/backends/NAPARI_CURATOR.md`; STRATEGY Trust paragraph. Governs U2–U4.

### High-Level Technical Design

```text
STRATEGY Trust ──► ADR 0014 claim matrix (red/green cells)
                         │
                         ▼
              language gate (R5)
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   Streamlit radio    Qt/napari docs   browser chrome
   (sole Trust name)  (demoted claims) (no Trust badge
                                        when degraded)
```

Sequence: write ADR + index → relabel UI/docs → degraded chrome → tests. No pipeline or Apply behavior changes.

### Risks & Dependencies

- **Risk:** Operators equate removing “MATLAB-style” from desktop labels with removing the feature. Mitigation: keep launch paths; clarify “desktop / experimental review” in help text.
- **Risk:** ADR cells stay red forever if no follow-on evidence plan. Mitigation: ADR Consequences section names follow-on ideation survivors (keymap, golden suite, `.m` restore) as the evidence path.
- **Dependency:** None on live MATLAB `.m` sources for this change set.

### Open Questions

**Deferred (non-blocking)**

- DQ1. Exact Trust badge UI affordance in the React curator (title chip vs status line) — implementer chooses the smallest chrome change that satisfies R3.
- DQ2. Whether `docs/TODO.md` gets a checkbox linking ADR 0014 — optional hygiene, not DoD.

---

## Implementation Units

### U1. ADR 0014 curator Trust claim matrix

**Goal:** Record the claim surface and checkable matrix as the language-gate authority.

**Requirements:** R4, R5, R7

**Files:** `docs/adr/0014-curator-trust-claim-matrix.md` (create), `docs/adr/README.md` (index row)

**Approach:** Follow ADR 0013 shape (In short, Status, Context, Decision, Consequences, Considered Options). Decision names browser as sole Trust claim surface and lists matrix cells with initial red/green. State that science Certification (0011/0012) is a separate Trust track. Ban `1:1` / identical phrasing until all claim-surface cells are green.

**Patterns:** `docs/adr/0013-claimed-energy-trace-provenance.md`, `docs/adr/README.md`

**Test scenarios:**

- None (docs-only unit). Verification is human review that the matrix lists every R4 cell.

**Verification:** ADR linked from README; cells match R4 list.

---

### U2. Streamlit curation workflow labels

**Goal:** Operator chooser names one Trust surface and honest desktop options.

**Requirements:** R1, R2, R6

**Files:** `slavv_python/interface/streamlit/views/curation.py`

**Approach:** Rename radio options and help text. Browser remains the Trust / MATLAB-familiar path (allowed naming per A4/KTD2). Desktop option drops “(MATLAB-style)”. Update any branch comparisons that hard-code the old option strings.

**Patterns:** existing radio + help in `curation.py` (~L76–121)

**Test scenarios:**

- See U6.

**Verification:** Manual glance at Curation page option strings; U6 green.

---

### U3. Demote Qt/napari over-claims

**Goal:** Remove “1:1” / Trust MATLAB-familiar marketing from desktop curator surfaces.

**Requirements:** R2, R5

**Files:** `slavv_python/visualization/interactive_curator.py`, `docs/reference/backends/NAPARI_CURATOR.md`

**Approach:** Rewrite `interactive_curator.py` module docstring to describe desktop GCI goals without “1:1 feature parity.” Adjust NAPARI doc so it does not imply Trust MATLAB-familiar or four-panel claim for napari; keep experimental framing.

**Patterns:** existing honest experimental wording already partially in NAPARI doc

**Test scenarios:**

- See U6 (forbidden-substring scan includes `interactive_curator.py` docstring region).

**Verification:** U6 green; no `1:1` in curator module docstring.

---

### U4. Workflow docs + STRATEGY pointer

**Goal:** Maintained docs and strategy name one Trust surface and cite ADR 0014.

**Requirements:** R6, R7

**Files:** `docs/reference/workflow/MANUAL_CURATION_WORKFLOW.md`, `STRATEGY.md`

**Approach:** Update “Curate the result” copy: browser = Trust MATLAB-familiar path; desktop = desktop/experimental review (no MATLAB-style Trust claim). Add one Trust-track sentence pointing at ADR 0014. Do not change science Certification wording beyond clarifying the GUI claim is separate.

**Patterns:** current MANUAL_CURATION sections; STRATEGY Trust / equivalence paragraph

**Test scenarios:**

- None automated beyond optional U6 doc path scan if included.

**Verification:** Docs read-through; STRATEGY cites `docs/adr/0014-curator-trust-claim-matrix.md`.

---

### U5. Degraded-mode Trust chrome suppression

**Goal:** When `degradedReason` is set, browser curator chrome does not show Trust MATLAB-familiar claim language.

**Requirements:** R3

**Files:** `slavv_python/interface/streamlit/components/matlab_curator/frontend/src/App.tsx` (and rebuild artifact if the package requires committing `frontend/build/`)

**Approach:** Gate any Trust / MATLAB-familiar title or badge on `!data.degradedReason`. Keep existing `<em>{degradedReason}</em>` disclosure. Prefer minimal JSX change; rebuild frontend per existing matlab_curator package scripts if build output is tracked.

**Patterns:** status line already renders `degradedReason` (~App.tsx L819)

**Test scenarios:**

- Prefer a small pure helper (e.g. `trustClaimVisible(degradedReason)`) with unit tests in U6 if extracting avoids brittle React DOM tests. Otherwise document manual AE3 check in Verification Contract.

**Verification:** U6 helper tests or manual degraded fixture check; AE3 satisfied.

---

### U6. Claim-language regression tests

**Goal:** Lock forbidden strings and workflow option naming so claims cannot silently return.

**Requirements:** R1, R2, R5, R6

**Files:** `tests/unit/interface/test_curation_trust_claim_labels.py` (create; adjust path if `tests/README.md` placement differs)

**Approach:**

1. Assert Streamlit workflow option constants / returned labels: browser option contains Trust or MATLAB-familiar naming; desktop option does not contain `MATLAB-style` or `1:1`.
2. Assert `interactive_curator.py` source does not contain `1:1` in the module docstring.
3. If U5 extracts a helper, assert `trustClaimVisible(None) is True` and `trustClaimVisible("…") is False`.

**Patterns:** nearest `tests/unit/interface/` Streamlit unit tests; `tests/README.md` placement rules

**Test scenarios:**

- Desktop option string rejects `MATLAB-style` and `1:1`.
- Browser option remains the sole Trust/MATLAB-familiar chooser label among the two manual review options.
- Qt module docstring rejects `1:1`.
- Degraded helper (if present) suppresses Trust claim when reason set.

**Verification:** `uv run pytest tests/unit/interface/test_curation_trust_claim_labels.py -q`

---

## Verification Contract

```powershell
uv run pytest tests/unit/interface/test_curation_trust_claim_labels.py -q
uv run ruff check slavv_python/interface/streamlit/views/curation.py slavv_python/visualization/interactive_curator.py
```

If frontend sources change and build output is committed:

```powershell
# follow package scripts under slavv_python/interface/streamlit/components/matlab_curator/
```

Manual: open Curation workflow radio (AE1); load a degraded Energy-only session and confirm no Trust badge (AE3); confirm Apply still available without prove-exact (AE4 — no code change expected).

---

## Definition of Done

- ADR 0014 exists, indexed in `docs/adr/README.md`, and defines all R4 cells with initial statuses per KTD1.
- Streamlit desktop option is not labeled MATLAB-style; browser is the named Trust MATLAB-familiar surface.
- Qt module docstring and NAPARI/workflow docs no longer assert 1:1 / Trust MATLAB-familiar for desktop paths.
- STRATEGY Trust paragraph cites ADR 0014.
- Degraded browser sessions suppress Trust claim chrome (R3).
- U6 tests green.
- Abandoned experimental copy from this change set is not left in the diff.
- No changes to Apply / Network rebuild / prove-exact gates (R7).
