"""LiveCodeBench scorer: extract code, run full public+private tests via the vendored evaluator."""
import json
from pathlib import Path

from .lcb_eval.code_generation import CodeGenerationProblem
from .lcb_eval.compute_code_generation_metrics import codegen_metrics

PROBLEMS_FILE = Path(__file__).resolve().parent.parent / "data" / "lcb_problems.jsonl"


def extract_code(model_output: str) -> str:
    """Content of the last ```...``` fenced block (matches lcb_runner default extraction)."""
    lines = (model_output or "").split("\n")
    fences = [i for i, ln in enumerate(lines) if "```" in ln]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1: fences[-1]])


def _load_problems():
    with open(PROBLEMS_FILE) as f:
        return {json.loads(l)["uid"]: json.loads(l)["row"] for l in f}


def score_lcb(pairs, timeout=6, num_process_evaluate=16):
    """pairs: list of (uid, model_output). Returns {uid: bool} pass@1 (all tests pass)."""
    rows = _load_problems()
    uids, samples, gens = [], [], []
    for uid, output in pairs:
        if uid not in rows:
            continue
        prob = CodeGenerationProblem(**rows[uid])
        uids.append(uid)
        samples.append(prob.get_evaluation_sample())
        gens.append([extract_code(output)])
    if not uids:
        return {}
    _, results, _ = codegen_metrics(
        samples, gens, k_list=[1], timeout=timeout,
        num_process_evaluate=num_process_evaluate, debug=False,
    )
    out = {}
    for idx, uid in enumerate(uids):
        gen_results = results.get(idx, [[]])[0]
        out[uid] = bool(gen_results) and all(r == True for r in gen_results)  # noqa: E712
    return out
