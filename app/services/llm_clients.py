import httpx
from app.config import settings


async def call_primary(system_prompt: str, user_text: str) -> str:
    """
    Optional: implement OpenAI (or other) here.
    For MVP, we keep it simple: if primary_provider == "none", raise to trigger fallback.
    """
    if settings.primary_provider != "openai":
        raise RuntimeError("Primary provider disabled")

    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    # Minimal OpenAI Responses API call over HTTP (no SDK dependency)
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.openai_model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # Extract text output (best-effort)
    out = []
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out.append(c.get("text", ""))
    return "\n".join(out).strip() or "(no output)"


async def call_fallback_ollama(system_prompt: str, user_text: str) -> str:
    """
    Requires: Ollama running locally (ollama serve) and model pulled (e.g., ollama pull llama3)
    """
    url = f"{settings.ollama_base_url}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return (data.get("message", {}) or {}).get("content", "").strip() or "(no output)"
