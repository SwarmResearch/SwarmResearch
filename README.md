# SwarmResearch

[![Blog](assets/blog_badge.svg)](https://yuvrajvirk.github.io/blogs/swarmresearch/swarmresearch.html)
[![Paper](assets/paper_badge.svg)](https://github.com/SwarmResearch/SwarmResearch/blob/main/assets/swarmresearch_paper.pdf)

*SwarmResearch* is a set of skills to orchestrate a population of independent research agents. It's great for letting coding agents loose on open-ended exploration tasks for hours and days without collapsing onto any one direction. 

| Vanilla autoresearch | SwarmResearch |
| --- | --- |
| One agent makes slow progress refining one idea  | Concurrent agents quickly screen new ideas before long-tail refinement |
| Keeping all improvements in one file biases agent towards small, greedy edits on one approach | All major edits are made in separate git branches making high-level changes easier |
| Accumulating context windows constrain exploration | Explorer subagents have fresh context windows while orchestrator steers strategic swarm behavior |

## How to use  
The swarm's working directory requires:
- `prompt.md`: file describing the problem and how to access the evaluator. This is used by subagents for context on the task.
- `evaluator`: A way an agent can score their solution. Coding agents are excellent reward hackers. We suggest keeping evaluator code outside of the agent's working directory and explicit instructions in `prompt.md` to not look for it.
  - SwarmResearch skills can be adapted for use without an explicit evaluator. Instead, the orchestrator can judge solution quality. For example, for open-ended data analysis, subagents can analyze data from diverse perspectives while the orchestrator judges and maintains their quality + diversity
- `baseline` (optional): A minimal working baseline working on your evaluator gets the agent writing successful solutions faster

For an example, check out the [speculative decoding setup](spec_dec_example/spec-dec-swarm-eval).

We recommend using `/goal` e.g. "/goal Follow the swarmresearch skill. Your search agent budget is 150. Keep going until you completely exhaust it. Your task and objective is in prompt.md."

The existing skills are for Claude Code and Codex, but they can easily extended for any coding agent. The only change is in the `shepherd` skill, which specifies how to spawn non-interactive agent sessions since it differs between coding agents. 

**Managing Cost:** As a reference, a 50 agent ~3 hour run on speculative decoding with Codex GPT-5.5 High used up 7% of my weekly limits on ChatGPT Pro 5x (May 26, 2026). Compared to vanilla autoresearch, you can get farther in less time since agents test multiple ideas concurrently and don't get stuck in local optima.

## Extending and editing the skills
The skills are simple and adapting them is easy:
- `shepherd`: The orchestrator. Spawns search agents and steers populations's search behavior. Prompts emphasize initializing and maintaining a diverse population of ideas, prioritizing effort on promising ideas, and breaking out of plateaus. It has access to 3 steering mechanisms when spawning new search agents: parent selection, search agent type, and prompts.
- `shepherd-explorer`: Explorers have fresh context windows and use the content in their worktree as context. Their goal is to explore new approaches. 
- `shepherd-optimizer`: Optimizers fork their parent agent's conversation history, in addition to using worktree content as context. Their goal is to make a few refinements to the parent solution.

The Shepherd Agent spawns Explorers and Optimizers by launching non-interactive Codex/CC sessions. This is necessary to support forking conversation histories for optimizers, which isn't possible with native subagents as far as I know. For a more complete description, read the blog or paper.

**Looking to reproduce experiments in paper?** [Reproduction repo here](https://github.com/SwarmResearch/swarmresearch-paper-reproduce)
