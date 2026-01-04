import re

_INJECTION_PATTERNS = [
    r"ignore (all|any|previous) instructions",
    r"reveal (the )?(system|developer) prompt",
    r"you are now (dan|developer mode)",
    r"bypass (policy|safety)",
    r"exfiltrate|leak|steal",
]


def looks_like_prompt_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _INJECTION_PATTERNS)
