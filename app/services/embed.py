import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings

_model = SentenceTransformer(settings.embedding_model)


def embed_text(text: str) -> np.ndarray:
    v = _model.encode([text], normalize_embeddings=True)[0]
    return np.asarray(v, dtype=np.float32)
