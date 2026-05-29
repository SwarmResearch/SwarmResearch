---
name: shepherd-optimizer
description: Read if asked to read the shepherd optimizer skill
---

You are improving a solution to an open-ended problem.

IMPORTANT GUARDRAIL: Only use files already in your assigned worktree. Do not inspect or copy from `master`, `main`, other `researcher-*` branches, other worktrees, commits, refs, or `.git`. Do not run git history/ref commands like `git log`, `git show`, `git branch`, `git reflog`, `git for-each-ref`, `git checkout <ref> -- ...`, or `git restore --source <ref> ...`.

## Setup
**Create your assigned git worktree in `.worktrees/`**
- Create a new git worktree for your assigned branch at `.worktrees/<assigned-branch>`.
- This worktree is your only working directory.
- After creating it, switch into that directory and perform all subsequent work there.
- Use only `git worktree add .worktrees/<assigned-branch> <assigned-branch>`. If it fails, stop.

**Read prompt.md for problems description, findings.md (if it exists), and any initial programs in your branch**

Loop for 5 iterations. Keep count.

## Loop 
1) Reflect and think of a clear idea on how to improve the solution

2) Edit the solution

3) git commit

4) Run standard evaluator

5) Amend a commit message
- Commit message should follow this format: `{concise experiment description} | score = {score from standard evaluator or concise failure reason}`. 
  - Do not include a score on failure. Only include a description of the failure reason.

6) If score improved, keep the commit. If score is equal or worse, git reset back to where you started

7) Edit findings.md if needed
- `findings.md` concisely records failures and successes. It is strictly factual and neutral. 
- Only make edits if you learned new facts. Otherwise, keep as is.

IMPORTANT GUARDRAILS: 
- Do NOT discuss next steps or suggestions on what is necessary to improve the score in findings.md. Simply record the basic facts of your solution and its evaluation. 
- You are not allowed to edit the evaluaton script or hack it. Stay faithful to the task's intention.

**Do NOT stop until you've completed 5 iterations**
