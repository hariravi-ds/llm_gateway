"""
Cache hits are risky. We verify equivalence before reuse.

MVP approach:
- Use a cross-encoder if available (more accurate)
- Else: do a stricter embedding similarity check + basic numeric token checks
"""
import re
from dataclasses import dataclass

try:
    from sentence_transformers import CrossEncoder
    _ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    _ce = None

_NUM_RE = re.compile(r"\b\d+(\.\d+)?\b")


def _numbers(text: str) -> set[str]:
    return set(_NUM_RE.findall(text))


@dataclass
class VerifyResult:
    ok: bool
    score: float


def verify_equivalence(query: str, cached_question: str) -> VerifyResult:
    # If numeric tokens differ, often not equivalent (simple guard)
    if _numbers(query) != _numbers(cached_question):
        return VerifyResult(False, 0.0)

    if _ce is not None:
        score = float(_ce.predict([(query, cached_question)])[
                      0])  # higher = more relevant
        # Threshold tuned by you; 0.7 is a reasonable starting point
        return VerifyResult(score >= 0.7, score)

    # Fallback: lexical overlap heuristic
    q = set(query.lower().split())
    c = set(cached_question.lower().split())
    jacc = len(q & c) / max(1, len(q | c))
    return VerifyResult(jacc >= 0.6, jacc)
