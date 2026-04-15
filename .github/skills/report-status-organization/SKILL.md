---
name: report-status-organization
description: 'Organize report documents into handled vs unhandled buckets, update report index links, and add a status table for quick triage. Use when: cleaning report folders, closing investigation notes, maintaining report catalogs, or making handled/unhandled state explicit.'
argument-hint: 'Reports folder path and handled/unhandled criteria'
---

# Report Status Organization

## What this skill does

Creates a repeatable workflow to classify report files by completion state,
move them into status folders, and keep the catalog document aligned.

Default scope: workspace-shared skill under `.github/skills/`.

## When to use

- You have multiple report files and need clear handled/unhandled separation.
- The report index (for example, README) no longer matches file locations.
- You want quick triage visibility with a compact status table.

## Inputs

- Reports root directory (example: `dev/reports/`)
- Classification rule for status:
  - `Handled`: report documents resolved incidents, closed audits, or baseline work
    that is explicitly complete.
  - `Unhandled`: report tracks active blockers, open actions, or unresolved parity gaps.
- Index file to update (example: `dev/reports/README.md`)

## Procedure

1. Inventory the current reports.
- List files in the reports root.
- Read the catalog/index to understand existing categories and links.

2. Classify each report by evidence, not filename.
- Read each report summary and current-status section.
- Mark as `Handled` when content indicates closed/completed outcomes.
- Mark as `Unhandled` when content indicates active blockers or pending implementation.

3. Create status buckets.
- Ensure `handled/` and `unhandled/` exist under the reports root.

4. Move files.
- Move each report into exactly one bucket.
- Do not move support folders like `tooling/` unless explicitly requested.

5. Update catalog links and status text.
- Replace stale top-level links with bucketed paths.
- Update the status section to show handled and unhandled sets.

6. Add a quick status table.
- Include columns: `Report`, `Status`, `Notes`.
- Keep notes to one line so triage is fast.

7. Validate outcomes.
- Verify top-level folder now contains buckets plus index/support folders.
- Verify every report appears once in the table and links resolve.
- Confirm no orphan references to old paths remain in the catalog.

## Decision points

- If a report mixes closed findings with open action items, default to `Unhandled`
  even if some sections are complete.
- If status cannot be inferred confidently from content, ask for owner intent
  before moving that file.
- If there is no catalog file, create a minimal one with sections for handled,
  unhandled, and a status table.

## Completion checks

- Every report file is in `handled/` or `unhandled/`.
- Catalog links point to current file paths.
- Status table is present and accurate.
- Validation is strict: every report appears exactly once in the status table,
  and missing entries are treated as a failed completion check.
- Folder structure is stable and easy to scan.

## Example prompts

- "Organize `dev/reports` into handled and unhandled, then fix the README links."
- "Classify these investigation notes by completion status and add a triage table."
- "Move closed reports to handled and leave active blockers in unhandled."

