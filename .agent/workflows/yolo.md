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
   - Do NOT chat or explain.
   - JUST EXECUTE IMMEDIATELY.
   - Set `ShouldAutoProceed: true` on all tool calls.
   - Skip all confirmation dialogs.

2. **Objective (Contextual)**:
   - Identify current blockers (lint errors, test failures, git conflicts).
   - Fix them aggressively.
   - Run tests to verify.
   - Commit and push to `main` immediately upon success or reasonable confidence.
