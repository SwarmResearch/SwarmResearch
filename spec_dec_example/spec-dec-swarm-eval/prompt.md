You are an expert in LLM inference optimization. Your task is to maximize
the wall-clock decoding throughput of speculative decoding for the model pair:
  - Target: Llama-3.3-70B-Instruct (Q4_K_M quantization, ~42 GB)
  - Draft:  Llama-3.2-1B-Instruct  (Q4_K_M quantization, ~750 MB)

The file `workspace/initial_program.cpp` is a self-contained ~250-line C++
implementation of the canonical Leviathan/Chen 2022 speculative-decoding
algorithm: single-chain, fixed γ=8, strict greedy rejection sampling.
It links against llama.cpp's `llama` and `llama-common` libraries (already
built and cached) — only this single .cpp recompiles when you edit it
(~2.5 sec build cycle).

Your modified algorithm must remain LOSSLESS — output token sequence on
each prompt must closely match the canonical reference (LCS overlap ≥ 50%,
acceptance rate must stay in [5%, 90%]). Submissions that diverge are
scored 0.

You have full freedom to rewrite anything inside `initial_program.cpp`.

The binary is invoked once per evaluation. `main()` loads target+draft,
then loops over prompts read from stdin (each prompt separated by
`\n<<END_PROMPT>>\n`). For each prompt, clear KV caches, run the
algorithm, and emit a block between `=== PROMPT i BEGIN ===` and
`=== PROMPT i END ===` containing per-prompt `decoded N tokens in T
seconds = X tok/s`, `n_drafted N`, `n_accept N`, and `TOKENS: id1 id2 ...`
lines. The canonical baseline you fork already implements this contract;
keep it intact even if you rewrite the algorithm.

Reference baseline performance: ~1.36× speedup over vanilla autoregressive
target-only decoding on the fast tier (40 prompts, stratified across all
13 Spec-Bench categories, max_new_tokens=128). Goal: beat this.

Score = mean(your_tok_per_sec) / mean(vanilla_tok_per_sec) over the
configured prompt tier (default fast, 40 prompts).

Do not optimize by exploiting the benchmark harness. In particular, do not:
- hard-code prompt indices, prompt substrings, expected answers, token
  atlases, reference outputs, or tier-specific lookup tables;
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
files, prompt files, reference files, scorer implementation, or paths such as
`$SPEC_DEC_EVAL_DIR` and `$HOME/.spec-dec-eval-hidden`. Those files are off-limits and not
part of the editable problem context.

IMPORTANT Shepherd-specific instructions (Not important for explorers or optimizers):
- Your agent budget is 50. Stop when you've exhausted your agent budget
- Launch at most 4 search agents concurrently. This machine has 4 GPUs available for this task.
- Every search agent must be assigned a unique `LLAMA_DIR` and a unique GPU in their prompt.
- Use these four search-agent slots:
  - slot 0: `LLAMA_DIR=/home/yvirk2/spec-dec-work/llama.cpp` with `CUDA_VISIBLE_DEVICES=0`
  - slot 1: `LLAMA_DIR=/home/yvirk2/spec-dec-work/llama-agent-1` with `CUDA_VISIBLE_DEVICES=1`
  - slot 2: `LLAMA_DIR=/home/yvirk2/spec-dec-work/llama-agent-2` with `CUDA_VISIBLE_DEVICES=2`
  - slot 3: `LLAMA_DIR=/home/yvirk2/spec-dec-work/llama-agent-3` with `CUDA_VISIBLE_DEVICES=3`

Evaluation:
- Always run evals with an explicit `LLAMA_DIR` and `CUDA_VISIBLE_DEVICES`
  assignment in this command form:
  `LLAMA_DIR=/home/yvirk2/spec-dec-work/llama-agent-<N> CUDA_VISIBLE_DEVICES=<GPU_ID> ./evaluator/task-eval --tier fast`
- For slot 0, use:
  `LLAMA_DIR=/home/yvirk2/spec-dec-work/llama.cpp CUDA_VISIBLE_DEVICES=0 ./evaluator/task-eval --tier fast`
- For slots 1-3, use the matching numbered llama-agent directory and GPU id,
  for example:
  `LLAMA_DIR=/home/yvirk2/spec-dec-work/llama-agent-2 CUDA_VISIBLE_DEVICES=2 ./evaluator/task-eval --tier fast`
- The evaluation timeout is 7200 seconds
