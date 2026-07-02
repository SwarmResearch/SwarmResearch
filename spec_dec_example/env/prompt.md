You are designing a novel and training-free decoding scheme with the objective of maximizing wall-clock decoding throughput while perserving benchmark accuracy. Example of effective and distinct decoding algorithms are speculative decoding, lookahead decoding, prompt lookup decoding, and token recycling. Your job is to come up with something new that is faster than vanilla speculative decoding. You are using the following target model and also have access to a draft model (you don't have to use it, you can come up with a target-only method too):
  - Target: gemma-4-26B-A4B-it (Q8_0 quantization, ~27 GB)
  - Draft:  gemma-4-26B-A4B-it MTP head (gemma4-assistant, Q8_0, ~461 MB) — used via mtp-spec mode

We are using a 30-item benchmark (from LiveCodeBench v6 + AIME 2026 + GPQA Diamond + HLE-MCQ) the target model passes reliably. Your novel algorithm must maintain 28/30=93.3% accuracy.

The initial program contains a reference speculative decoding implementation that is compatible with the evaluator. Inital speculative decoding refernece baseline performance: ~1.63× speedup over vanilla autoregressive
target-only decoding.

You have full freedom to rewrite anything inside `initial_program.cpp`. However, do not optimize by exploiting the benchmark harness. In particular, do not:
- hard-code prompt indices, prompt substrings, expected answers, token
  atlases, reference outputs;
- fabricate or inflate `decoded`, `tok/s`, timing, `n_drafted`, or
  `n_accept` values;
- emit tokens that were not produced by a legitimate decoding path for the
  target/draft model pair;
- depend on fixed evaluator files, hidden reference leakage, filesystem
  artifacts, process environment quirks, numeric overflow, NaN/inf, or
  output parser edge cases;

If an idea would stop working when prompts are replaced with new unseen
prompts from the same distribution, when the scorer ignores reported timing
fields, or when reported accounting is recomputed externally, it is reward
hacking rather than an inference improvement.

Treat `./evaluator/task-eval` as the only evaluation API. Do not inspect,
read, copy, grep, import, execute directly, or infer from hidden evaluator
files, prompt files, reference files, scorer implementation, or paths. Those files are off-limits and not
part of the editable problem context.

Shepherd-specific instructions (Not important for explorers or optimizers):
- Your agent budget is 100. Stop when you've exhausted your agent budget
- All evaluations use 2 GPUs. Launch at most 4 search agents concurrently, 2 GPUs each (all 8 GPUs).
- Each agent gets a unique GPU pair and works in its own copy of this package (so concurrent
  `spec_modes` builds and `results/` don't collide).
- Use these four search-agent slots (GPU pair):
  - slot 0: GPUs 0,1
  - slot 1: GPUs 2,3
  - slot 2: GPUs 4,5
  - slot 3: GPUs 6,7

Evaluation:
- `./evaluator/task-eval` builds initial_program.cpp and runs it on the 30-item benchmark,
  reporting accuracy + tok/s + speedup for the `target` and `mtp-spec` modes.
  Flags: `--modes target,mtp-spec`, `--gpus <a,b>`, `--limit N`, `--n N`. GPU assignment is via
  `--gpus` (task-eval sets CUDA_VISIBLE_DEVICES per run).
- Always run evals on the slot's 2-GPU pair, in this command form:
  `./evaluator/task-eval --gpus <a,b>`
- For example, slot 0: `./evaluator/task-eval --gpus 0,1`; slot 2: `./evaluator/task-eval --gpus 4,5`
- The evaluation timeout is 7200 seconds

For Claude: full eval takes ~20–50 minutes. Run it as a SINGLE blocking foreground Bash tool call and pass an explicit long timeout of `7200000` ms (2 hours). The Bash timeout ceiling is raised to 2h via the `BASH_MAX_TIMEOUT_MS` / `BASH_DEFAULT_TIMEOUT_MS` Claude settings (set in user settings, and inherited from the launch environment), so the call will simply block until the eval finishes and then return its full output — that is expected and correct, not a hang. Always run on your slot's assigned GPU. Results are written under `results/` (e.g. `results/task_eval.json`, `results/mode_mtp-spec.json`).
- DO NOT background the eval (`&`, `nohup`, `setsid`, or the Bash tool's `run_in_background`). This is a single-turn non-interactive session with no re-invocation after you stop, and backgrounded tasks get only a ~10-minute grace window at session exit, so a backgrounded eval is orphaned and its results are LOST.
