from fastapi import FastAPI
from fastapi.responses import Response
from app.models import ChatRequest, ChatResponse
from app.config import settings
from app.middleware.safety import looks_like_prompt_injection
from app.middleware.pii import redact_pii
from app.middleware.observability import (
    REQ_TOTAL, CACHE_HIT, CACHE_MISS, PRIMARY_CALL, FALLBACK_CALL, BLOCKED,
    timed, metrics_response
)
from app.services.embed import embed_text
from app.services.cache import cache_lookup, cache_store, make_sys_hash
from app.services.retrieval import retrieve_docs
from app.services.verifier import verify_equivalence
from app.services.llm_clients import call_primary, call_fallback_ollama
from app.services.rag import build_rag_prompt

app = FastAPI(title="Production RAG Gateway (Cache + Firewall + Fallback)")


@app.get("/metrics")
def metrics():
    data, ctype = metrics_response()
    return Response(content=data, media_type=ctype)


@app.post("/v1/chat", response_model=ChatResponse)
@timed
async def chat(req: ChatRequest):
    REQ_TOTAL.inc()

    threshold = req.cache_threshold if req.cache_threshold is not None else settings.cache_threshold

    # --- Safety gate: prompt injection check
    if looks_like_prompt_injection(req.user_text):
        BLOCKED.inc()
        return ChatResponse(
            answer="Request blocked by safety gate (suspected prompt injection).",
            from_cache=False,
            model_used="blocked",
            safety_blocked=True,
        )

    # --- PII redaction
    pii = redact_pii(req.user_text)
    user_text = pii.redacted_text

    # --- Embed once (used for cache + retrieval)
    qvec = embed_text(user_text)
    sys_hash = make_sys_hash(req.system_prompt)

    # --- FAST PATH: semantic answer cache + equivalence verification
    hit, meta, sim = cache_lookup(
        tenant_id=req.tenant_id,
        policy_version=req.policy_version,
        sys_hash=sys_hash,
        doc_version=req.doc_version,
        query_vec=qvec,
        threshold=threshold,
    )

    if hit:
        vr = verify_equivalence(user_text, hit.get("question", ""))
        if vr.ok:
            CACHE_HIT.inc()
            return ChatResponse(
                answer=hit["answer"],
                from_cache=True,
                model_used="cache",
                cache_similarity=sim,
                pii_redacted=pii.redacted,
                citations=(meta or {}).get("citations"),
            )

    CACHE_MISS.inc()

    # --- SLOW PATH: standard RAG over document store
    chunks = retrieve_docs(
        tenant_id=req.tenant_id,
        doc_version=req.doc_version,
        query_vec=qvec,
        top_k=settings.retrieve_top_k,
    )

    sys_prompt, augmented_user, citations = build_rag_prompt(
        req.system_prompt, user_text, chunks)

    # --- Primary model, fallback on failure
    try:
        PRIMARY_CALL.inc()
        answer = await call_primary(sys_prompt, augmented_user)
        model_used = "primary"
    except Exception:
        FALLBACK_CALL.inc()
        answer = await call_fallback_ollama(sys_prompt, augmented_user)
        model_used = "fallback_ollama"

    # --- Write-through cache (store verified slow-path result)
    cache_store(
        tenant_id=req.tenant_id,
        policy_version=req.policy_version,
        sys_hash=sys_hash,
        doc_version=req.doc_version,
        question=user_text,
        answer=answer,
        vec=qvec,
        meta={
            "citations": citations,
            "pii_redacted": pii.redacted,
            "model_used": model_used,
        },
    )

    return ChatResponse(
        answer=answer,
        from_cache=False,
        model_used=model_used,
        cache_similarity=sim,
        pii_redacted=pii.redacted,
        citations=citations,
    )
