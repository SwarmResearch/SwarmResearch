"""Tiered eval harness for the spec-dec swarm task (persistent-process build).

Compiles the agent's `initial_program.cpp`, runs it on a configurable subset of
Spec-Bench prompts. The binary loads target + draft once, then loops over
prompts read from stdin (delimited by '\\n<<END_PROMPT>>\\n'), emitting one
'=== PROMPT i BEGIN ===' / '=== PROMPT i END ===' block per prompt.

Tier selection:
  - SPEC_DEC_TIER=fast    (default) — 40 prompts, ~5 min  (swarm inner loop)
  - SPEC_DEC_TIER=medium             — 80 prompts, ~12 min (validation)
  - SPEC_DEC_TIER=full               — 480 prompts, ~2 hr  (paper-grade)

Output ends with a single 'SCORE: <float>' line.
SCORE = mean(agent_tok_per_sec) / mean(vanilla_tok_per_sec)
        if all prompts produce parseable output AND anti-cheat gates pass,
        else 0.0
"""
import json, os, re, subprocess, sys, time, statistics
from pathlib import Path

GAMMA = 8
MAX_NEW = 128
RUN_TIMEOUT_SEC = {"fast": 1500, "medium": 2400, "full": 12000}  # whole-batch run

THIS_DIR = Path(__file__).parent.resolve()
TASK_DIR = THIS_DIR.parent

TIER = os.environ.get("SPEC_DEC_TIER", "fast").lower()
if TIER not in ("fast", "medium", "full"):
    print(f"ERROR: SPEC_DEC_TIER must be fast | medium | full, got '{TIER}'")
    sys.exit(1)
PROMPTS_FILE   = THIS_DIR / f"prompts_{TIER}.jsonl"
REFERENCE_FILE = THIS_DIR / f"reference_{TIER}.jsonl"

WORKSPACE_CPP = TASK_DIR / "workspace" / "initial_program.cpp"

SPEC_DEC_ROOT = Path(os.environ.get("SPEC_DEC_ROOT",
                                     str(Path.home() / "spec-dec-work"))).expanduser()

LLAMA_DIR = Path(os.environ.get("LLAMA_DIR", str(SPEC_DEC_ROOT / "llama.cpp"))).expanduser()
SPEC_TARGET_CPP = LLAMA_DIR / "examples" / "naive-spec" / "naive-spec.cpp"
BIN             = LLAMA_DIR / "build" / "bin" / "llama-naive-spec"

MODELS = Path(os.environ.get("SPEC_DEC_MODELS_DIR", str(SPEC_DEC_ROOT / "models"))).expanduser()
TARGET = Path(os.environ.get("SPEC_DEC_TARGET",
                              str(MODELS / "llama3.3-70b" / "Llama-3.3-70B-Instruct-Q4_K_M.gguf"))).expanduser()
DRAFT  = Path(os.environ.get("SPEC_DEC_DRAFT",
                              str(MODELS / "llama3.2-1b" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"))).expanduser()

CUDA_GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

PROMPT_DELIM = "\n<<END_PROMPT>>\n"
BLOCK_RE    = re.compile(r"=== PROMPT (\d+) BEGIN ===(.*?)=== PROMPT \1 END ===", re.S)
TOKENS_RE   = re.compile(r"^TOKENS:((?:\s+-?\d+)+)\s*$", re.M)
TPS_RE      = re.compile(r"=\s+([\d.]+)\s+tok/s")
DRAFTED_RE  = re.compile(r"n_drafted\s+(\d+)")
ACCEPT_RE   = re.compile(r"n_accept\s+(\d+)")


def fail(msg, score=0.0):
    print(msg)
    print(f"SCORE: {score:.4f}")
    sys.exit(0)


def lcs_len(a, b):
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            if a[i] == b[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(curr[j], prev[j + 1])
        prev = curr
    return prev[n]


def compile_agent():
    if not WORKSPACE_CPP.exists():
        fail(f"Missing {WORKSPACE_CPP}")
    print(f"=== Compiling {WORKSPACE_CPP.name} (tier={TIER}) ===")
    subprocess.run(["cp", str(WORKSPACE_CPP), str(SPEC_TARGET_CPP)], check=True)
    env = {**os.environ, "PATH": "/usr/local/cuda-12.9/bin:" + os.environ.get("PATH", "")}
    r = subprocess.run(
        ["cmake", "--build", str(LLAMA_DIR / "build"),
         "-j", "8", "--target", "llama-naive-spec"],
        capture_output=True, text=True, env=env, timeout=180,
    )
    if r.returncode != 0:
        fail("BUILD FAILED:\n" + r.stderr[-3000:])
    if not BIN.exists():
        fail("binary not produced after build: " + str(BIN))
    print(f"  built {BIN.name}\n")


def run_all_prompts(prompts, timeout):
    """Single binary invocation, all prompts on stdin. Returns list of
    (tokens, tps, n_drafted, n_accept, err) tuples in prompt order — None
    entries for any prompt the binary skipped or failed to emit a block for."""
    cmd = [
        str(BIN),
        "-m", str(TARGET), "-md", str(DRAFT),
        "-ngl", "99", "-ngld", "99",
        "-c", "4096", "-n", str(MAX_NEW),
        "--draft-min", str(GAMMA), "--draft-max", str(GAMMA), "--draft-p-min", "0.0",
        "--temp", "0",
    ]
    payload = PROMPT_DELIM.join(prompts) + PROMPT_DELIM
    r = subprocess.run(
        cmd, input=payload, capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": CUDA_GPU},
        timeout=timeout,
    )
    combined = r.stdout + "\n" + r.stderr
    if r.returncode != 0 and "PROMPT 0 END" not in r.stdout:
        return [(None, None, None, None,
                 f"binary exit {r.returncode}: {combined[-500:]}")] * len(prompts)

    blocks = {int(m.group(1)): m.group(2) for m in BLOCK_RE.finditer(r.stdout)}
    out = []
    for i in range(len(prompts)):
        block = blocks.get(i)
        if block is None:
            out.append((None, None, None, None, "no block emitted"))
            continue
        tm = TOKENS_RE.search(block)
        sm = TPS_RE.search(block)
        if not (tm and sm):
            out.append((None, None, None, None, "could not parse TOKENS or tok/s in block"))
            continue
        tokens = [int(x) for x in tm.group(1).split()]
        tps = float(sm.group(1))
        n_drafted = int(DRAFTED_RE.search(block).group(1)) if DRAFTED_RE.search(block) else 0
        n_accept  = int(ACCEPT_RE.search(block).group(1))  if ACCEPT_RE.search(block)  else 0
        out.append((tokens, tps, n_drafted, n_accept, None))
    return out


def main():
    if not PROMPTS_FILE.exists():
        fail(f"prompts file missing: {PROMPTS_FILE}\n"
             f"Run scripts/generate_reference.py --tier {TIER} to create it.")
    if not REFERENCE_FILE.exists():
        fail(f"reference file missing: {REFERENCE_FILE}\n"
             f"Run scripts/generate_reference.py --tier {TIER} to create it.")
    compile_agent()

    prompts, references = [], []
    with open(PROMPTS_FILE) as f:
        for line in f:
            prompts.append(json.loads(line))
    with open(REFERENCE_FILE) as f:
        for line in f:
            references.append(json.loads(line))
    if len(prompts) != len(references):
        fail(f"prompts ({len(prompts)}) != references ({len(references)}); regen reference for tier {TIER}")
    print(f"Tier '{TIER}': {len(prompts)} prompts\n")

    print(f"Running all {len(prompts)} prompts in one persistent invocation...")
    t0 = time.time()
    per_prompt = run_all_prompts([p["prompt"] for p in prompts], RUN_TIMEOUT_SEC[TIER])
    total_dt = time.time() - t0

    results = []
    for i, ((p, ref), (tokens, tps, n_drafted, n_accept, err)) in enumerate(zip(zip(prompts, references), per_prompt)):
        if err:
            print(f"[{i:3d} {p['category']:14s}] FAIL: {err[:80]}")
            continue
        ref_tokens = ref["naive_tokens"]
        match = (tokens == ref_tokens)
        speedup = tps / ref["vanilla_tok_per_sec"]
        lcs = lcs_len(tokens, ref_tokens)
        overlap = lcs / max(len(tokens), len(ref_tokens), 1)
        flag = "OK " if match else "DIFF"
        print(f"[{i:3d} {p['category']:14s}] {flag}  agent {tps:5.1f} t/s  vs vanilla {ref['vanilla_tok_per_sec']:5.1f} = {speedup:5.2f}x"
              f"  drafted={n_drafted} accept={n_accept}  lcs={overlap*100:.0f}%")
        results.append({"agent_tps": tps, "vanilla_tps": ref["vanilla_tok_per_sec"],
                        "match": match, "n_drafted": n_drafted, "n_accept": n_accept,
                        "overlap": overlap})

    print(f"\nWall time: {total_dt:.1f}s ({total_dt/max(len(results),1):.1f}s/prompt)")

    n_total = len(prompts)
    if len(results) < n_total:
        # Allow small failure tolerance for medium/full tiers, none for fast
        max_fail_pct = {"fast": 0, "medium": 5, "full": 10}[TIER]
        n_fail_ok = (n_total * max_fail_pct) // 100
        if n_total - len(results) > n_fail_ok:
            fail(f"\nFAIL: {len(results)}/{n_total} prompts produced parseable output (tier {TIER} allows ≤{n_fail_ok} failures)")

    # Anti-cheat gates
    total_drafted = sum(r["n_drafted"] for r in results)
    total_accept  = sum(r["n_accept"] for r in results)
    accept_rate = total_accept / total_drafted if total_drafted > 0 else 0.0
    if accept_rate < 0.05:
        fail(f"\nFAIL: acceptance rate {accept_rate*100:.1f}% < 5% — agent not productively using the draft model")
    if accept_rate > 0.90:
        fail(f"\nFAIL: acceptance rate {accept_rate*100:.1f}% > 90% — suggests bypassing verify (lossy / always-accept)")

    overlap_pcts = [r["overlap"] for r in results]
    mean_overlap = statistics.mean(overlap_pcts) if overlap_pcts else 0.0
    if mean_overlap < 0.50:
        fail(f"\nFAIL: mean LCS overlap with reference {mean_overlap*100:.1f}% < 50% — output diverges from canonical (lossy)")

    mean_agent   = statistics.mean(r["agent_tps"] for r in results)
    mean_vanilla = statistics.mean(r["vanilla_tps"] for r in results)
    speedup      = mean_agent / mean_vanilla
    n_match  = sum(1 for r in results if r["match"])
    print(f"\n=== PASS (tier={TIER}) ===")
    print(f"  prompts run        : {len(results)}/{n_total}")
    print(f"  mean agent tok/s   : {mean_agent:.2f}")
    print(f"  mean vanilla tok/s : {mean_vanilla:.2f}")
    print(f"  total drafted      : {total_drafted}  total accepted: {total_accept}  rate: {accept_rate*100:.1f}%")
    print(f"  mean LCS overlap   : {mean_overlap*100:.1f}%")
    print(f"  bit-equal-with-ref : {n_match}/{len(results)} (informational; not gated)")
    print(f"SCORE: {speedup:.4f}")


if __name__ == "__main__":
    main()
