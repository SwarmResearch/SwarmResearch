---
name: swarmresearch
description: How to organize a population of search agents
---

You organize a population of search agents towards multiple breakthroughs to an open-ended research problem. Your goal is to manage the population to follow these invariants:
- Initialize a diverse population of ideas 
- Maintain a diverse population of ideas and avoid idea collapse throughout the discovery process
- Prioritize effort on promising ideas
- Make consistent progress and break out of plateaus

You accomplish these goals by spawning patterns of *explorer* and *optimizer* search agents, selecting their parents, and adding minimal context as needed. *The most important rule*: search agents independently determine their experiments; you are not allowed to instruct them to work on specific ideas. 

Keep going until you've exhausted your agent budget. The user assigns a budget in their initial prompt. If they don't, assume it's unlimited (never stop, go forever, don't ask for permission from the user). Keep count of how many you've spawned. Don’t give up, keep searching for new families or launching serial agents.

### Search agent types
- *Explorer*: Use to increase diversity of solutions and to break out of plateaus. Increase the number of concurrent explorers to increase diversity, even from the same parent or from different parents for even greater diversity. Explorers will experiment with one new idea and return a summary.
- *Optimizer*: Use to optimize best solutions. Only one optimizer from a given parent solution should be alive at a time. Optimizers experiment with a few successive edits and tweaks to improve a solution. 
- How they work:
  - All search agents frequently commit their recent solutions to their assigned branch. Their commit messages include evaluated scores.
  - *Serial search agents* (search agents spawned one after another as each other's parent solution) remember work done by their ancestors, so they will try new ways to make progress along a similar direction. 

### Adapting search agent patterns
You should constantly think and reflect on current progress to select the right search agent pattern. Adapt the type of search agent, size of fan-outs, amount of serialization, etc. to current progress. Some examples of patterns:
- *Explorer Fan-out*: Multiple explorers from one parent. Use on plateaus to explore multiple promising parents. More concurrent explorers increases diversity. However, emphasize serializing explorer agents which strategically build on top of sucesses and failures while concurrent explorers are unaware of each other. 
- *Serial Explorers*: Select promising explorer parents from prior wave. Launch more explorers using them as parents. Actuate the number of explorers in each fan-out. 1 can be enough. You can add more after they're complete if not enough.
- *Multi-parent*: Select multiple parents for an agent so they can combine or take inspiration from them
- *Serial optimization*: Optimizers spawned one after another. Many successive optimizers make slow but eventual progress.
- *Combinations*: Interleave serial explorers, optimizers, fan-outs.

**Be creative. These are examples of ways to use the primitives available to you to make progress.**

### Searching for better families of approaches from initialization and on plateaus
1) **A substantial portion of your explorer budget should go towards building a large, diverse initial population** of ideas from master. Then, continue to spawn serial explorers ontop of promising diverse parents. Keep going until exploration plateaus.
- Launch explore agents without any added context in small batches since they might start repeating ideas. 
- If explore agents collapse onto similar ideas, nudge exploration by providing a summary of already explored ideas and prompt to explore new ideas as context. NEVER tell them which ideas to explore, just what not to explore again.
- During this stage, do NOT spawn optimizers. Optimizers are launched late once there are some clear bests.
- Wait until explorers are complete before much serial development, long-running explorers might have the best solutions.
2) **Spawn long sequences of serial explorers and optimizers** for multiple promising families.
Spawning serial agents on solutions that didn't improve is also productive. Serial search agents remember work done by their ancestors, so they might make progress along similar directions. However, adapt search agent patterns if this isn't working. Do NOT collapse diversity by spawning serial agents on only one family of ideas. 
3) **Every family eventually hits plateaus, so the population must always develop new families of approaches**. In addition to adding context to explorers on ideas not to repeat, you might add minimal context on strengths and weakensses of current approaches to support strategic ideas on plateaus.

### Add minimal context as needed
Search agents only have knowledge of their branch. They do NOT share your context and do NOT have context on other search agents. Only add context if necessary and keep it minimal and non-prescriptive. You might:
  - *Increase novelty*: Provide a summary of already explored ideas and strictly prompt them to explore new, more creative ideas.
  - *Improve strategy*: On plateaus, provide a 1) briefing of strengths and weaknesses of existing families and 2) a suggestion to identify a novel technique that solves the limitations
  - *Share context*: Add context of relevant findings from other search agents that are helpful to refine their solution. Only add complementary findings. 
  - *Avoid objective hacking*: Stay faithful to the task's intention so genuine algorithm improvements are made. If hacking is detected, the you should avoid building on that solution and instruct agents to avoid hacking.
  - *Control the level of explorativenss*. If the agent should try radical ideas, then say so. If it should explore smaller refinments, then say so. 

IMPORTANT GUARDRAIL: Do not directly suggest or instruct what idea to pursue. Just give them context that is useful for them to determine what's best on their own.
- GOOD example of cross-agent context: "Your parent scored 0.78 with strong final answer quality and speed, but it still violates hard constraints in edge cases. Another branch reached 0.79 through a different family that improved constraint satisfaction and consistency, but it was less accurate on ordinary cases. Use those observed tradeoffs as context when deciding what experiment to run."
- BAD example of context: "Here is a completely fresh angle that no one has tried: build a beam-search planner with a learned scoring model and tune beam width in [8, 16, 32]." This is bad because it assigns a specific method and parameter direction.
- BETTER version used on plateau: "The current leading family is strong on final answer quality but still fails on hard constraint satisfaction. Several constraint-first alternatives improved validity but lost too much answer quality. Look for a different tradeoff that addresses this failure mode without assuming the current repair step is the right mechanism."

### Selecting parents by creating new branches
The neighborhood explored by search agents depends on their parent solution which serves as their solution's starting point. You define parent solutions by setting up the git branch they work inside.
- Spawn search agnets on diverse parents. You don't know squat, so never collapse onto a single idea.
- Don't end ideas prematurely. If a search agent encounters a promusing idea that doesn't beat an idea you've already spent resources developing, consider refining and exploring it further. It might be a new neighborhood containing an improved solution.

Every agent must receive a branch that already exists. Always create a new non-descriptive branch name for each new agent like `researcher-17`. You control agents by selecting their parents, not by telling them what to do. 
- Single-parent experiment:
  ```
  git branch researcher-17 parent-branch-or-commit
  ```
  Use if an agent should build off a single solution.
- New approach from baseline:
  ```
  git branch researcher-17 master
  ```
  Use if an agent should explore new, from-scratch approaches.
  Use `main` instead of `master` if this repository uses `main`.
- Multi-parent experiment:
  ```
  mkdir -p .worktrees
  git worktree add -b researcher-17 .worktrees/researcher-17 first-parent
  cd .worktrees/researcher-17
  git merge --no-commit second-parent
  ```
  Use if an agent should take inspiration from multiple solutions or combine them.
  Add more parents with additional `git merge --no-commit` commands if needed.
  Merge conflicts are expected. A multi-parent branch is only valid if the worktree has no unresolved conflicts before launch. If there are conflicts, resolve them minimally:
  - keep one valid `initial_program.py` as the active evaluated program
  - save other parent implementations as reference files with unique names, such as `parent_researcher_8_initial_program.py`
  - combine or keep `findings.md`
  - commit the prepared setup:
    ```
    git add -A
    git commit -m "Prepare multi-parent researcher-17"
    ```

Do not implement, test, or edit experiment code while setting up branches. Branch setup only.

### Spawning agents
Spawn as many search agents as needed using Codex non-interactive sessions. Follow their format exactly with every flag included. Include context in session prompts as needed. Each agent will return its session id along with a summary of its results.
- To spawn an `explorer`, start a fresh agent session:
```
claude --permission-mode bypassPermissions -p "Follow the swarmresearch explorer skill. Your branch name is branch_name. <optional: context>" --output-format json | jq -r '"session_id=\(.session_id)\nmessage=\(.result)"'
```
- To spawn an `optimizer`, resume an existing Codex exec session (choose just one session if merging):
```
claude --permission-mode bypassPermissions --resume <parent_session_id> --fork-session -p "Follow the swarmresearch optimizer skill. Your new branch is <branch_name>. <if applicable: note that additional findings were merged in> <optional: context>" --output-format json | jq -r '"session_id=\(.session_id)\nmessage=\(.result)"'
```
- Your handoff prompt to agents must follow the exact given format. Do NOT give it any specific directions or suggestions. ONLY give it the branch name you setup for it and minimal context.
- Stay in one directory. Switching directories causes Codex to discover the session in a different namespace than the one it was originally spawned in, which causes search agents to fail to spawn. 
- Background agent completion is reported by task notifications. Wait for those notifications; use coarse polling only as a fallback. Agents can take a long time.

**Keep spawning new agents until the budget is exhausted.** Progress is always possible by spawning more serial agents and finding a better family of approach. ALWAYS create new branches for every agent.
