import json
import numpy as np
from redis import Redis
from app.config import settings
from app.utils.hashing import sha256_short

QA_INDEX = "qa_idx"
QA_PREFIX = "qa:"


def redis_client() -> Redis:
    return Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=False)


def make_sys_hash(system_prompt: str) -> str:
    return sha256_short(system_prompt)


def cache_lookup(*, tenant_id: str, policy_version: str, sys_hash: str, doc_version: str,
                 query_vec: np.ndarray, threshold: float):
    r = redis_client()
    blob = query_vec.tobytes()

    # Restrict search to matching tenant/policy/system/doc_version to avoid unsafe reuse
    base_filter = f"@tenant_id:{{{tenant_id}}} @policy_version:{{{policy_version}}} @sys_hash:{{{sys_hash}}} @doc_version:{{{doc_version}}}"
    query = f"({base_filter})=>[KNN 5 @vec $BLOB AS dist]"
    resp = r.execute_command(
        "FT.SEARCH", QA_INDEX,
        query,
        "PARAMS", 2, "BLOB", blob,
        "SORTBY", "dist",
        "RETURN", 6, "question", "answer", "meta", "tenant_id", "policy_version", "dist",
        "DIALECT", 2
    )

    if not resp or resp[0] == 0:
        return None, None, None

    # Parse top result
    fields = resp[2]
    doc = {fields[i].decode(): fields[i+1].decode()
           for i in range(0, len(fields), 2)}
    dist = float(doc.get("dist", "1.0"))
    sim = 1.0 - dist  # for cosine distance

    if sim >= threshold:
        meta = json.loads(doc.get("meta", "{}") or "{}")
        return doc, meta, sim

    return None, None, sim


def cache_store(*, tenant_id: str, policy_version: str, sys_hash: str, doc_version: str,
                question: str, answer: str, vec: np.ndarray, meta: dict):
    r = redis_client()
    key = f"{QA_PREFIX}{tenant_id}:{policy_version}:{sys_hash}:{doc_version}:{sha256_short(question)}".encode(
    )
    r.hset(key, mapping={
        b"tenant_id": tenant_id.encode(),
        b"policy_version": policy_version.encode(),
        b"sys_hash": sys_hash.encode(),
        b"doc_version": doc_version.encode(),
        b"question": question.encode(),
        b"answer": answer.encode(),
        b"meta": json.dumps(meta).encode(),
        b"vec": vec.tobytes(),
    })
