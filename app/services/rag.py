from typing import List, Dict, Any


def build_rag_prompt(system_prompt: str, user_text: str, chunks: List[Dict[str, Any]]) -> tuple[str, list]:
    citations = []
    context_lines = []
    for i, ch in enumerate(chunks, start=1):
        context_lines.append(f"[{i}] {ch['text']}")
        citations.append({
            "doc_id": ch.get("doc_id"),
            "chunk_id": ch.get("chunk_id"),
            "similarity": ch.get("similarity"),
        })

    augmented_user = (
        "Use the following context to answer. If the context is insufficient, say so.\n\n"
        "Context:\n" + "\n".join(context_lines) + "\n\n"
        f"User question: {user_text}"
    )

    return system_prompt, augmented_user, citations
