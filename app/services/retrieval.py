import numpy as np
from redis import Redis
from app.config import settings

DOC_INDEX = "doc_idx"


def redis_client() -> Redis:
    return Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=False)


def retrieve_docs(*, tenant_id: str, doc_version: str, query_vec: np.ndarray, top_k: int):
    r = redis_client()
    blob = query_vec.tobytes()

    base_filter = f"@tenant_id:{{{tenant_id}}} @doc_version:{{{doc_version}}}"
    query = f"({base_filter})=>[KNN {top_k} @vec $BLOB AS dist]"
    resp = r.execute_command(
        "FT.SEARCH", DOC_INDEX,
        query,
        "PARAMS", 2, "BLOB", blob,
        "SORTBY", "dist",
        "RETURN", 6, "doc_id", "chunk_id", "text", "dist", "tenant_id", "doc_version",
        "DIALECT", 2
    )

    hits = []
    if not resp or resp[0] == 0:
        return hits

    # resp: [total, doc_key1, [fields...], doc_key2, [fields...], ...]
    for i in range(1, len(resp), 2):
        doc_key = resp[i]
        fields = resp[i+1]
        d = {fields[j].decode(): fields[j+1].decode()
             for j in range(0, len(fields), 2)}
        d["redis_key"] = doc_key.decode() if isinstance(
            doc_key, (bytes, bytearray)) else str(doc_key)
        d["dist"] = float(d.get("dist", "1.0"))
        d["similarity"] = 1.0 - d["dist"]
        hits.append(d)

    return hits
