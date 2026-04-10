# Test Organization

Keep tests under `tests/` organized by the owning package surface, not by the
historical task that introduced them.

## Placement Rules

- `tests/unit/<owner>/` for package behavior owned by a source package such as
  `analysis`, `utils`, `io`, `core`, or `apps`
- `tests/unit/parity/` for pure-Python parity helpers, comparison layout, and
  comparison runtime logic
- `tests/diagnostic/` for environment checks, MATLAB availability, and
  cross-runtime parity harnesses

## Notes

- Prefer moving a misfiled test into the matching owner directory instead of
  reshaping production code around the old location.
- Keep regression intent in the test body and markers; keep directory names
  about ownership.
- Preserve public imports when they are part of the compatibility surface, even
  if the underlying implementation lives in a more specific module.
