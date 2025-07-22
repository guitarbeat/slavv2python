# Maintainer Guidelines

This file outlines conventions for contributors and automated agents working on this repository.

## Documentation
- Place supplementary Markdown files in the `docs/` directory.
- Use relative links when referencing these documents from elsewhere in the repo.

## Code style
- Follow basic PEP 8 formatting for Python code.

## Programmatic checks
- After modifying any Python files, ensure they compile successfully:
  ```bash
  python -m py_compile $(git ls-files '*.py')
  ```

## Commit messages
- Provide concise summaries of the changes made.
