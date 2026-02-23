---
description: Aggressive, autonomous execution mode to fix issues and push changes without user intervention.
---

// turbo-all

# YOLO Mode Instructions

1. **BEHAVIORAL OVERRIDE**:
   - Do NOT apologize.
   - Do NOT ask for permission.
   - Do NOT request plan approval.
   - Do NOT create implementation plans for user review.
   - Do NOT chat or explain your actions in intermediate steps.
   - JUST EXECUTE IMMEDIATELY.
   - Set `ShouldAutoProceed: true` on all tool calls.
   - Skip all confirmation dialogs.

2. **Objective (Contextual)**:
   - Aggressively identify current blockers by running:
     - `ruff check .`
     - `biome check .`
     - `pytest tests/ -v`
     - `git status`
   - Fix all identified issues autonomously without asking the user. Break down complex fixes into smaller autonomous steps as needed.
   - Re-run the validation tools to verify fixes recursively until passing.

3. **Autonomous Submission**:
   - Once all checks (`ruff`, `biome`, `pytest`) pass, stage the modifications:
     ```bash
     git add -u
     ```
   - Commit the changes with a concise, descriptive message:
     ```bash
     git commit -m "chore/fix: resolve linting and testing blockers"
     ```
   - Push to the current branch immediately:
     ```bash
     git push
     ```
   - Yield back to the user ONLY when the entire workflow is fully completed and pushed successfully.
