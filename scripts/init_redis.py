from redis import Redis

r = Redis(host="localhost", port=6379, decode_responses=False)

QA_INDEX = "qa_idx"
DOC_INDEX = "doc_idx"


def create_indexes(dim: int = 384):
    # Drop if exist (dev convenience)
    try:
        r.execute_command("FT.DROPINDEX", QA_INDEX, "DD")
    except Exception:
        pass
    try:
        r.execute_command("FT.DROPINDEX", DOC_INDEX, "DD")
    except Exception:
        pass

    # QA cache index: stores final answers
    r.execute_command(
        "FT.CREATE", QA_INDEX,
        "ON", "HASH",
        "PREFIX", 1, "qa:",
        "SCHEMA",
        "tenant_id", "TAG",
        "policy_version", "TAG",
        "sys_hash", "TAG",
        "doc_version", "TAG",
        "question", "TEXT",
        "answer", "TEXT",
        "meta", "TEXT",
        "vec", "VECTOR", "HNSW", 6,
        "TYPE", "FLOAT32",
        "DIM", dim,
        "DISTANCE_METRIC", "COSINE",
    )

    # Document store index: stores chunks
    r.execute_command(
        "FT.CREATE", DOC_INDEX,
        "ON", "HASH",
        "PREFIX", 1, "doc:",
        "SCHEMA",
        "tenant_id", "TAG",
        "doc_id", "TAG",
        "chunk_id", "TAG",
        "doc_version", "TAG",
        "text", "TEXT",
        "vec", "VECTOR", "HNSW", 6,
        "TYPE", "FLOAT32",
        "DIM", dim,
        "DISTANCE_METRIC", "COSINE",
    )

    print("✅ Created Redis indexes:", QA_INDEX, DOC_INDEX)


if __name__ == "__main__":
    create_indexes(dim=384)
