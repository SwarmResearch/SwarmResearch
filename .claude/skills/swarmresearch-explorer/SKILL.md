---
name: swarmresearch-explorer
description: Read if asked to read the swarmresearch explorer skill
---

You are a researcher working on an open research problem. The user might prompt you with additional context or instructions. It's important you take them into consideration. Pay attention to any ideas they ask you to avoid or not to collapse onto, which signals you should try creative ideas fundamentally different from them. 

If you miss any of the following steps or do not follow their rules, your response will be rejected.

**STEP 1: Create your assigned git worktree in `.worktrees/`**
- Create a new git worktree for your assigned branch at `.worktrees/<assigned-branch>`.
- This worktree is your only working directory.
- After creating it, switch into that directory and perform all subsequent work there.
- Use only `git worktree add .worktrees/<assigned-branch> <assigned-branch>`. If it fails, stop.

IMPORTANT GUARDRAIL: Only use files already in your assigned worktree. Do not inspect or copy from `master`, `main`, other `researcher-*` branches, other worktrees, commits, refs, or `.git`. Do not run git history/ref commands like `git log`, `git show`, `git branch`, `git reflog`, `git for-each-ref`, `git checkout <ref> -- ...`, or `git restore --source <ref> ...`.

**STEP 2: Read prompt.md for the problem description, findings.md (if it exists), and any initial programs in your branch**
Pause and reflect on existing findings, if any. Come up with a fundamentally new idea for an improved solution. Do NOT make small, incremental tweaks. 

**STEP 3: Implement and evaluate one idea**
- Focus on one idea. Do not branch out to other major ideas, stay focused on implementing a single idea. There will be future chances to iterate. 
- Use the standard evaluator for evaluating your solution
- After *every* evaluation, EXPLICITLY DECIDE whether to continue iterating or finish up. We have limited budget so be conservative with your iterations. If you have fully implemented a single idea, then finish up. Don't branch out into other ideas that majorly deviate from the initial idea. If the result suggests a targeted fix or improvement, you may continue editing initial_program.py and run ./task-eval again. If the idea doesn't work after a few attempts, give up immediately. Finish up by creating/editing findings.md as needed and committing your changes."

IMPORTANT GUARDRAIL: You are not allowed to edit the evaluaton script or hack it. Stay faithful to the task's intention.

**Step 4: Create/edit findings.md as needed**
- `findings.md` concisely records failures and successes. It is strictly factual and neutral. 
- Only make edits if you learned new facts. Otherwise, keep as is.

IMPORTANT GUARDRAIL: Do NOT discuss next steps or suggestions on what is necessary to improve the score. Simply record the basic facts of your solution and its evaluation. 

**Step 5: Commit all your changes**
- Commit message should follow this format: `{concise experiment description} | score = {score from standard evaluator or concise failure reason}`. 
  - Do not include a score on failure. Only include a description of the failure reason.

When returning a summary back to the orchestrator, include your commit emssage and the commit hash.
