"""Deterministic scorers for AIME (boxed integer) and MCQ (letter) outputs."""
import re

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_INT = re.compile(r"-?\d+")
_ANSWER_LINE = re.compile(r"(?:final\s+answer|answer)\s*[:\-]?\s*\**\s*([A-J])\b", re.IGNORECASE)
_BOXED_LETTER = re.compile(r"\\boxed\{\s*([A-J])\s*\}")


def score_aime(output: str, gold: str) -> bool:
    """Gold is an integer 0-999. Prefer the last \\boxed{...}; fall back to last integer."""
    try:
        g = int(str(gold).strip())
    except ValueError:
        return False
    boxed = _BOXED.findall(output or "")
    if boxed:
        nums = _INT.findall(boxed[-1])
        if nums:
            return int(nums[-1]) == g
    nums = _INT.findall(output or "")          # fallback: last integer mentioned
    return bool(nums) and int(nums[-1]) == g


def score_mcq(output: str, gold: str) -> bool:
    """Gold is a single letter (A-J). Prefer an explicit 'Answer: X' line, then \\boxed{X},
    then the last standalone letter."""
    g = str(gold).strip().upper()
    out = output or ""
    m = list(_ANSWER_LINE.finditer(out))
    if m:
        return m[-1].group(1).upper() == g
    b = _BOXED_LETTER.findall(out)
    if b:
        return b[-1].upper() == g
    # fallback: last standalone capital letter token A-J
    toks = re.findall(r"\b([A-J])\b", out)
    return bool(toks) and toks[-1].upper() == g


SCORERS = {"aime": score_aime, "mcq": score_mcq}
